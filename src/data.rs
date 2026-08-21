//! Minimal `.npy` reader (no dependency). Realization corpora are plain
//! little-endian C-order `(N, 3, 3)` f64 arrays of monodromy triples `[C, G, T]`.
//! Optional label arrays, such as `linspace_strata.npy`, are `(N,)` integers.
use std::fs;

/// Parse the `.npy` header, returning `(shape, data_offset)`. Asserts C-order.
fn header(bytes: &[u8]) -> (Vec<usize>, usize) {
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "not a .npy file");
    let (hlen, hstart) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        v => panic!("unsupported .npy version {v}"),
    };
    let head = std::str::from_utf8(&bytes[hstart..hstart + hlen]).unwrap();
    assert!(head.contains("'fortran_order': False"), "need C-order .npy");
    let s = head.find("'shape':").unwrap();
    let open = head[s..].find('(').unwrap() + s + 1;
    let close = head[open..].find(')').unwrap() + open;
    let shape = head[open..close]
        .split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .collect();
    (shape, hstart + hlen)
}

/// Read an (N,3,3) f64 array as `N` flat `[f64; 9]` rows (`[C0..2, G0..2, T0..2]`).
pub fn read_triples(path: &str) -> Vec<[f64; 9]> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let (shape, off) = header(&bytes);
    assert_eq!(&shape[1..], &[3, 3], "expected (N,3,3), got {shape:?}");
    let n = shape[0];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = [0.0f64; 9];
        for (k, slot) in row.iter_mut().enumerate() {
            let p = off + (i * 9 + k) * 8;
            *slot = f64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
        }
        out.push(row);
    }
    out
}

/// Read an (N,) i8 array (the stratum labels).
pub fn read_labels(path: &str) -> Vec<i8> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let (shape, off) = header(&bytes);
    (0..shape[0]).map(|i| bytes[off + i] as i8).collect()
}
