"""Independent finite-output audit. Does not import prototype.py."""
import itertools
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PERMS = list(itertools.permutations(range(4)))

def match(a, b):
    distances = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :])
    errors = [distances[np.arange(4), p] for p in PERMS]
    chosen = min(range(24), key=lambda i: errors[i].sum())
    return {"sum_assignment_max": float(errors[chosen].max()),
            "optimal_bottleneck": float(min(e.max() for e in errors))}

def adj(a):
    return np.swapaxes(a.conj(), -1, -2)

def differentiation(grid):
    first = np.zeros((len(grid), len(grid)))
    second = first.copy()
    for i in range(1, len(grid)-1):
        offsets = grid[i-1:i+2] - grid[i]
        system = np.array([offsets**j for j in range(3)])
        first[i, i-1:i+2] = np.linalg.solve(system, [0, 1, 0])
        second[i, i-1:i+2] = np.linalg.solve(system, [0, 0, 2])
    return first, second

def differential_arrays(x, y, h):
    dx, dxx = differentiation(x)
    dy, dyy = differentiation(y)
    hx = np.einsum('ab,bcij->acij', dx, h)
    hy = np.einsum('ab,cbij->caij', dy, h)
    lap = np.einsum('ab,bcij->acij', dxx, h) + np.einsum('ab,cbij->caij', dyy, h)
    return hx, hy, lap, -np.diag(dxx)[:, None]-np.diag(dyy)[None, :]

def interpolate(x, y, field, z):
    i = int(np.clip(np.searchsorted(x, z.real)-1, 0, len(x)-2))
    j = int(np.clip(np.searchsorted(y, z.imag)-1, 0, len(y)-2))
    a = (z.real-x[i])/(x[i+1]-x[i])
    b = (z.imag-y[j])/(y[j+1]-y[j])
    return ((1-a)*(1-b)*field[i,j]+a*(1-b)*field[i+1,j]
            +(1-a)*b*field[i,j+1]+a*b*field[i+1,j+1])

def transport(x, y, connection, vertices, steps=128):
    value = np.eye(4, dtype=complex)
    for start, stop in zip(vertices[:-1], vertices[1:]):
        dz = (stop-start)/steps
        for k in range(steps):
            z = start+k*dz
            c0 = -dz*interpolate(x,y,connection,z)
            cm = -dz*interpolate(x,y,connection,z+dz/2)
            c1 = -dz*interpolate(x,y,connection,z+dz)
            k1 = c0@value
            k2 = cm@(value+k1/2)
            k3 = cm@(value+k2/2)
            k4 = c1@(value+k3)
            value += (k1+2*k2+2*k3+k4)/6
    return value

def frame_audit(o, data):
    d = np.diag(np.exp(1j*np.pi*np.array(data['a'])))
    b = np.diag(np.exp(2j*np.pi*np.array(data['b'])))
    target = np.exp(2j*np.pi*np.array(data['target']))
    return {"orthogonality": float(np.linalg.norm(o.T@o-np.eye(4))),
            "determinant": float(np.linalg.det(o)),
            "target_roots": match(np.linalg.eigvals(d@o@b@o.T@d), target)}

def main():
    data = json.loads((ROOT/'fixture.json').read_text())
    feasibility = json.loads((ROOT/'fixture-feasibility.json').read_text())
    output = {"fixture": frame_audit(np.array(feasibility['O']), data)}
    output['fixture']['norm_sum_over_pi'] = float(2*(max(abs(np.array(data['a'])))+max(abs(np.array(data['b'])))))
    output['fixture']['phase_spreads'] = {k: float(np.ptp(data[k])) for k in ['a','b','target']}
    output['fixture']['minimum_root_gaps'] = {}
    for k in ['a','b','target']:
        roots = np.exp(2j*np.pi*np.array(data[k]))
        output['fixture']['minimum_root_gaps'][k] = float(min(abs(roots[i]-roots[j]) for i in range(4) for j in range(i)))
    output['levels'] = []
    for path in sorted(ROOT.glob('target-*.json')):
        run = json.loads(path.read_text())
        level = run['level']
        saved = np.load(ROOT/f'metric-{level}.npz')
        x,y = saved['x'],saved['y']
        h = saved['h'].reshape(len(x),len(y),4,4)
        hx,hy,lap,scale = differential_arrays(x,y,h)
        p = hx+1j*hy
        residual = lap-p@np.linalg.solve(h, adj(p))
        xx,yy = np.meshgrid(x,y,indexing='ij')
        free = np.ones(xx.shape, bool)
        free[[0,-1],:] = False
        free[:,[0,-1]] = False
        e = run['epsilon']
        for center in [0,1]:
            free &= ~((abs(xx-center)<=e*(1+1e-9)) & (abs(yy)<=e*(1+1e-9)))
        normalized = float(np.linalg.norm(residual[free]/scale[free,None,None])/np.sqrt(free.sum()))
        item = {"level":level,"frame":frame_audit(np.array(run['O']),data),
                "metric_min_eigenvalue":float(np.linalg.eigvalsh(h).min()),
                "hermiticity":float(abs(h-adj(h)).max()),
                "real_symmetry":float(abs(h[:,::-1]-h.conj()).max()),
                "normalized_pde_residual":normalized,
                "reported_pde_residual":run['logs'][-1]['residual'],
                "unscaled_pde_rms":float(np.linalg.norm(residual[free])/np.sqrt(free.sum()))}
        boundary_error = 0.
        weights = [-np.array(data['a']),-np.array(data['b']),np.array(data['target'])]
        flags = np.array(data['flags'])
        for i,j in np.argwhere(~free):
            z = complex(x[i],y[j])
            if abs(z.real)<=e*(1+1e-9) and abs(z.imag)<=e*(1+1e-9):
                which, radius, power = 0, max(abs(z),e/2), 1
            elif abs(z.real-1)<=e*(1+1e-9) and abs(z.imag)<=e*(1+1e-9):
                which, radius, power = 1, max(abs(z-1),e/2), 1
            else:
                which, radius, power = 2, max(abs(z),e/2), -1
            expected = flags[which]@np.diag(np.exp(2*power*weights[which]*np.log(radius)))@flags[which].T
            boundary_error = max(boundary_error,float(abs(h[i,j]-expected).max()))
        item['boundary_metric_error'] = boundary_error
        connection = np.linalg.solve(h,(hx-1j*hy)/2)
        hb = interpolate(x,y,h,.5+0j).real
        eigenvalues,vectors = np.linalg.eigh(hb)
        sqrt = (vectors*np.sqrt(eigenvalues))@vectors.T
        invsqrt = (vectors/np.sqrt(eigenvalues))@vectors.T
        paths = [[.5,.3,.3+.3j,-.3+.3j,-.3-.3j,.3-.3j,.3,.5],
                 [.5,.7,.7-.3j,1.3-.3j,1.3+.3j,.7+.3j,.7,.5]]
        us = [sqrt@transport(x,y,connection,vertices)@invsqrt for vertices in paths]
        item['independent_peripheral'] = [
            {"roots":match(np.linalg.eigvals(u),np.exp(2j*np.pi*np.array(data[k]))),
             "unitarity":float(np.linalg.norm(adj(u)@u-np.eye(4))),
             "symmetry":float(np.linalg.norm(u-u.T))} for u,k in zip(us,['a','b'])]
        item['raw_holonomy_product_target'] = match(np.linalg.eigvals(us[0]@us[1]),np.exp(2j*np.pi*np.array(data['target'])))
        outer_path = [.5,.5-.8j,1.6-.8j,1.6+.8j,-.6+.8j,-.6-.8j,.5-.8j,.5]
        outer = sqrt@transport(x,y,connection,outer_path)@invsqrt
        doubled = [sqrt@transport(x,y,connection,vertices,steps=256)@invsqrt for vertices in paths+[outer_path]]
        item['transport_128_vs_256_max_matrix_difference'] = float(max(np.linalg.norm(u-v) for u,v in zip(us+[outer],doubled)))
        item['large_ccw_loop'] = {
            'target_roots':match(np.linalg.eigvals(outer),np.exp(2j*np.pi*np.array(data['target']))),
            'inverse_infinity_roots':match(np.linalg.eigvals(np.linalg.inv(outer)),np.exp(-2j*np.pi*np.array(data['target']))),
            'unitarity':float(np.linalg.norm(adj(outer)@outer-np.eye(4))),
            'minus_U0_U1':float(np.linalg.norm(outer-us[0]@us[1])),
            'minus_U1_U0':float(np.linalg.norm(outer-us[1]@us[0]))}
        output['levels'].append(item)
    # Smooth control is exactly flat: g=I+zN, h=g* g and det(g)=1.
    grid = np.linspace(-1,1,17)
    z = grid[:,None]+1j*grid[None,:]
    n = np.array([[0,.3,.1,0],[0,0,.25,.1],[0,0,0,.2],[0,0,0,0]])
    g = np.eye(4)+z[...,None,None]*n
    control = adj(g)@g
    hx,hy,lap,scale = differential_arrays(grid,grid,control)
    p = hx+1j*hy
    r = lap-p@np.linalg.solve(control,adj(p))
    output['manufactured_control'] = {
        'discrete_exact_metric_residual_max':float(abs(r[1:-1,1:-1]).max()),
        'min_eigenvalue':float(np.linalg.eigvalsh(control).min()),
        'note':'Validates exact manufactured metric, not a separately saved control solver output.'}
    evidence = ['prototype.py','fixture.json','fixture-feasibility.json','control.json']
    evidence += [p.name for p in sorted(ROOT.glob('target-*.json'))]
    evidence += [p.name for p in sorted(ROOT.glob('metric-*.npz'))]
    output['sha256'] = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in evidence}
    (ROOT/'independent-verifier-output.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__ == '__main__':
    main()
