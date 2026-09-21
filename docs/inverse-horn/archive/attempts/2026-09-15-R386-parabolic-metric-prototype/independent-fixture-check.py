"""Independent feasibility and quantum-cut audit; never imports the candidate."""
from pathlib import Path
import hashlib
import itertools
import json
import re
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
fixture = json.loads((HERE / 'fixture.json').read_text(), parse_float=F)
plant = json.loads((HERE / 'fixture-feasibility.json').read_text())
source = (ROOT / 'crates/core/src/horn.rs').read_text()
table = source.split('const GR2_PRODUCTS:')[1].split('];', 1)[0]
rows = []
for labels in ([8, 4, 2, 1], None, [14, 13, 11, 7]):
    if labels is None:
        for left, right, target, degree in re.findall(
                r'product\(0b([01]+), 0b([01]+), 0b([01]+), ([0-2])\)', table):
            i, j, k, d = int(left, 2), int(right, 2), int(target, 2), int(degree)
            rows.append((i, j, k, d))
            if i != j:
                rows.append((j, i, k, d))
    else:
        for i in range(4):
            for j in range(4):
                rows.append((labels[i], labels[j], labels[(i+j) % 4], int(i+j >= 4)))
assert len(rows) == 72
def total(mask, x):
    return sum((x[i] for i in range(4) if mask & (1 << i)), F(0))
def slack(row, x):
    i, j, k, d = row
    return d-total(i, a)-total(j, b)+total(k, x)
def bits(mask):
    return [int(bool(mask & (1 << i))) for i in range(4)]
a, b, t = (fixture[k] for k in ('a', 'b', 'target'))
slacks = [slack(row, t) for row in rows]
assert min(slacks) > 0
constraints, bounds = [], []
for i, j, k, d in rows:
    if d == 0:
        constraints.append([-v for v in bits(k)])
        bounds.append(float(-total(i, a)-total(j, b)))
for i in range(3):
    c = [0]*4
    c[i+1], c[i] = 1, -1
    constraints.append(c)
    bounds.append(0)
constraints.append([1, 0, 0, -1])
bounds.append(1)
best = None
for index, row in enumerate(rows):
    if row[3] == 0:
        continue
    result = linprog(bits(row[2]), A_ub=constraints, b_ub=bounds,
                     A_eq=[[1]*4], b_eq=[0], bounds=[(None, None)]*4, method='highs')
    assert result.success
    point = [F(float(x)).limit_denominator(1000000) for x in result.x]
    assert sum(point) == 0 and all(point[i] >= point[i+1] for i in range(3))
    assert point[0]-point[3] <= 1
    assert all(slack(r, point) >= 0 for r in rows if r[3] == 0)
    value = slack(row, point)
    if best is None or value < best[0]:
        best = value, index, row, point
assert best[0] < 0
O = np.array(plant['O'])
af, bf, tf = (np.array([float(x) for x in z]) for z in (a, b, t))
E = np.diag(np.exp(1j*np.pi*af))
U = E @ O @ np.diag(np.exp(2j*np.pi*bf)) @ O.T @ E
roots = np.linalg.eigvals(U)
expected = np.exp(2j*np.pi*tf)
error = min(max(abs(roots[i]-expected[j]) for i,j in enumerate(p))
            for p in itertools.permutations(range(4)))
assert np.linalg.norm(O.T@O-np.eye(4)) < 1e-12 and abs(np.linalg.det(O)-1) < 1e-12
assert error < 1e-12
output = {
    'fixture_sha256': hashlib.sha256((HERE/'fixture.json').read_bytes()).hexdigest(),
    'feasibility_sha256': hashlib.sha256((HERE/'fixture-feasibility.json').read_bytes()).hexdigest(),
    'horn_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
    'rows': len(rows), 'degree_counts': {str(d): sum(r[3] == d for r in rows) for d in range(3)},
    'minimum_slack': str(min(slacks)), 'minimum_slack_float': float(min(slacks)),
    'trace_residuals': [str(sum(z)) for z in (a,b,t)],
    'alcove_widths': [str(z[0]-z[-1]) for z in (a,b,t)],
    'all_rows': [{'masks': list(r[:3]), 'degree': r[3], 'slack': str(s)} for r,s in zip(rows, slacks)],
    'quantum_separation': {'row_index': best[1], 'masks': list(best[2][:3]), 'degree': best[2][3],
                           'point': [str(x) for x in best[3]], 'slack': str(best[0]),
                           'degree_zero_minimum': str(min(slack(r,best[3]) for r in rows if r[3]==0))},
    'chosen_tracezero_normsum_over_pi': str(2*(max(map(abs,a))+max(map(abs,b)))),
    'plant_orthogonality_frobenius': float(np.linalg.norm(O.T@O-np.eye(4))),
    'plant_determinant': float(np.linalg.det(O)), 'plant_root_error': float(error),
}
(HERE/'independent-fixture-check.json').write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps({k:v for k,v in output.items() if k != 'all_rows'}, indent=2))
