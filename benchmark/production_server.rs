//! JSON-lines adapter for the production solver, without diagnostic routes.
use can_sandwich::{Rung, solve};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

fn coordinates(request: &Value, key: &str) -> Result<[f64; 3], String> {
    let values = request[key]
        .as_array()
        .filter(|v| v.len() == 3)
        .ok_or_else(|| format!("{key} must contain three finite numbers"))?;
    let mut result = [0.0; 3];
    for (slot, value) in result.iter_mut().zip(values) {
        *slot = value
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(|| format!("{key} must contain three finite numbers"))?;
    }
    Ok(result)
}

fn respond(request: &Value) -> Result<Value, String> {
    let id = request["id"]
        .as_u64()
        .ok_or("id must be a nonnegative integer")?;
    let c = coordinates(request, "c")?;
    let g = coordinates(request, "g")?;
    let t = coordinates(request, "t")?;
    let solution = solve(c, g, t);
    if solution.rung == Rung::Unsolved {
        return Ok(json!({"id": id, "status": "declined"}));
    }
    if solution
        .o
        .iter()
        .any(|z| !z.re.is_finite() || !z.im.is_finite() || z.im != 0.0)
    {
        return Err("production solver returned a nonreal or nonfinite frame".into());
    }
    let o: [[f64; 4]; 4] = std::array::from_fn(|i| std::array::from_fn(|j| solution.o[(i, j)].re));
    Ok(json!({"id": id, "status": "solved", "o": o}))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    writeln!(
        stdout,
        "{}",
        json!({"protocol": "can-sandwich-v1", "ready": true})
    )?;
    stdout.flush()?;
    for line in stdin.lock().lines() {
        let request: Value = serde_json::from_str(&line?)?;
        let response = match respond(&request) {
            Ok(response) => response,
            Err(message) => json!({"id": request["id"], "status": "error", "message": message}),
        };
        writeln!(stdout, "{response}")?;
        stdout.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_request_returns_an_identity_frame() {
        let response =
            respond(&json!({"id": 7, "c": [0,0,0], "g": [0,0,0], "t": [0,0,0]})).unwrap();
        assert_eq!(response["id"], 7);
        assert_eq!(response["status"], "solved");
        let rows = response["o"].as_array().unwrap();
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|row| row.as_array().unwrap().len() == 4));
    }

    #[test]
    fn malformed_requests_are_rejected() {
        for request in [
            json!({"id": -1, "c": [0,0,0], "g": [0,0,0], "t": [0,0,0]}),
            json!({"id": 0, "c": [0,0], "g": [0,0,0], "t": [0,0,0]}),
            json!({"id": 0, "c": [0,"NaN",0], "g": [0,0,0], "t": [0,0,0]}),
        ] {
            assert!(respond(&request).is_err());
        }
    }
}
