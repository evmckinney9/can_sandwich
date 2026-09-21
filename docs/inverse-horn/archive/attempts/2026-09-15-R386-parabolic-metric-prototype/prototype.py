"""R386: finite-difference Chern-flat Dirichlet metric pilot; no endpoint fit."""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, LinearOperator, gmres
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent
BAS=[]
for i in range(4):
 b=np.zeros((4,4),complex); b[i,i]=1; BAS.append(b)
for i in range(4):
 for j in range(i):
  b=np.zeros((4,4),complex); b[i,j]=b[j,i]=1/np.sqrt(2); BAS.append(b)
  b=np.zeros((4,4),complex); b[i,j]=1j/np.sqrt(2); b[j,i]=-1j/np.sqrt(2); BAS.append(b)
BAS=np.array(BAS)
def pack(h): return np.einsum('aij,...ij->...a',BAS.conj(),h).real
def unpack(v): return np.einsum('...a,aij->...ij',v,BAS)
def adj(a): return a.conj().swapaxes(-1,-2)
def herm(a): return (a+adj(a))/2

def derivs(x):
 n=len(x); d=sparse.lil_matrix((n,n)); l=d.copy()
 for i in range(1,n-1):
  a=x[i]-x[i-1]; b=x[i+1]-x[i]
  d[i,i-1:i+2]=[-b/(a*(a+b)),(b-a)/(a*b),a/(b*(a+b))]
  l[i,i-1:i+2]=[2/(a*(a+b)),-2/(a*b),2/(b*(a+b))]
 return d.tocsr(),l.tocsr()

class Mesh:
 def __init__(self,x,y,holes):
  self.x,self.y=x,y; self.nx,self.ny=len(x),len(y)
  xx,yy=np.meshgrid(x,y,indexing='ij'); self.z=xx+1j*yy
  boundary=np.zeros(xx.shape,bool); boundary[[0,-1],:]=True; boundary[:,[0,-1]]=True
  for p,e in holes: boundary|=(abs(xx-p)<=e*(1+1e-9)) & (abs(yy)<=e*(1+1e-9))
  self.fixed=boundary.ravel(); self.free=np.flatnonzero(~self.fixed)
  dx,lx=derivs(x);dy,ly=derivs(y)
  self.dx=sparse.kron(dx,sparse.eye(len(y))).tocsr()
  self.dy=sparse.kron(sparse.eye(len(x)),dy).tocsr()
  self.lap=(sparse.kron(lx,sparse.eye(len(y)))+sparse.kron(sparse.eye(len(x)),ly)).tocsr()
  self.lu=splu(self.lap[self.free,:][:,self.free].tocsc())
 def op(self,op,h): return (op@h.reshape(-1,16)).reshape(-1,4,4)
 def solve(self,boundary,maxouter=30):
  h=boundary.reshape(-1,4,4).copy();h[self.free]=0
  h[self.free]=unpack(self.lu.solve(-pack(self.op(self.lap,h)[self.free])))
  scale=-self.lap.diagonal()[self.free]
  n=len(self.free)*16; logs=[]; start=time.perf_counter()
  for iteration in range(maxouter+1):
   inv=np.linalg.inv(h);px=self.op(self.dx,h);py=self.op(self.dy,h);p=px+1j*py
   r=herm(self.op(self.lap,h)-p@inv@adj(p))[self.free]
   rv=pack(r);err=np.linalg.norm(rv/scale[:,None])/np.sqrt(len(self.free))
   logs.append({'iteration':iteration,'residual':float(err),'min_eigenvalue':float(np.linalg.eigvalsh(h).min()),'seconds':time.perf_counter()-start})
   print(json.dumps(logs[-1]),flush=True)
   if err<1e-9: return h,logs,'converged'
   if iteration==maxouter:break
   def jac(v):
    dh=np.zeros_like(h);dh[self.free]=unpack(v.reshape(-1,16))
    dp=self.op(self.dx,dh)+1j*self.op(self.dy,dh)
    val=self.op(self.lap,dh)-dp@inv@adj(p)-p@inv@adj(dp)+p@inv@dh@inv@adj(p)
    return pack(herm(val)[self.free]).ravel()
   pre=LinearOperator((n,n),matvec=lambda v:self.lu.solve(v.reshape(-1,16)).ravel(),dtype=float)
   count=[0]
   def cb(_):count[0]+=1
   delta,info=gmres(LinearOperator((n,n),matvec=jac,dtype=float),-rv.ravel(),M=pre,rtol=1e-5,atol=1e-12,restart=20,maxiter=4,callback=cb,callback_type='pr_norm')
   dh=unpack(delta.reshape(-1,16));alpha=1.;accepted=False
   for _ in range(16):
    trial=h.copy();trial[self.free]+=alpha*dh
    if np.linalg.eigvalsh(trial).min()>1e-11:
     pt=self.op(self.dx,trial)+1j*self.op(self.dy,trial)
     rt=herm(self.op(self.lap,trial)-pt@np.linalg.inv(trial)@adj(pt))[self.free]
     et=np.linalg.norm(pack(rt)/scale[:,None])/np.sqrt(len(self.free))
     if et<(1-1e-4*alpha)*err:accepted=True;break
    alpha*=.5
   logs[-1].update(gmres_info=int(info),gmres_steps=count[0],alpha=alpha)
   if not accepted:return h,logs,'line_search_failed'
   h=trial
  return h,logs,'iteration_budget'

def manufactured(z):
 n=np.array([[0,.3,.1,0],[0,0,.25,.1],[0,0,0,.2],[0,0,0,0]])
 g=np.eye(4)+z[...,None,None]*n
 return adj(g)@g

def alcove(roots):
 a=np.sort(np.angle(roots)/(2*np.pi))[::-1]
 s=int(round(sum(a)))
 if s>0:a[:s]-=1
 if s<0:a[s:]+=1
 return np.sort(a)[::-1]

def make_fixture():
 rng=np.random.default_rng(386)
 a=np.array([.42,.09,-.17,-.34]);b=np.array([.39,.16,-.21,-.34])
 q,_=np.linalg.qr(rng.normal(size=(4,4)))
 if np.linalg.det(q)<0:q[:,0]*=-1
 roots=np.linalg.eigvals(np.diag(np.exp(2j*np.pi*a))@q@np.diag(np.exp(2j*np.pi*b))@q.T)
 t=alcove(roots)
 flags=[]
 for _ in range(3):
  f,_=np.linalg.qr(rng.normal(size=(4,4)));flags.append(f.tolist())
 fixture={'a':a.tolist(),'b':b.tolist(),'target':t.tolist(),'flags':flags,'generator_seed':386,'norm_sum_over_pi':float(2*(max(abs(a))+max(abs(b))))}
 (ROOT/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
 (ROOT/'fixture-feasibility.json').write_text(json.dumps({'O':q.tolist(),'note':'Verifier-only planted feasibility data; solver receives fixture.json only.'},indent=2)+'\n')
 print(json.dumps(fixture))

def target_mesh(level):
 count=[8,12,16][level];e=[.125,.0625,.03125][level];r=[2.,4.,8.][level]
 inner=np.geomspace(e,.5,count);outer=np.geomspace(e,r,count+3)
 x=np.unique(np.r_[-outer,0,inner,1-inner,1,1+outer]);y=np.unique(np.r_[-outer,0,outer])
 return Mesh(x,y,[(0,e),(1,e)]),e,r

def boundary_metric(mesh,data,e):
 z=mesh.z.ravel();frames=np.array(data['flags'])
 # Positive loops have exp(-2pi i mu). Use inverse spectral weights.
 mus=[-np.array(data['a']),-np.array(data['b']),np.array(data['target'])]
 h=np.empty((len(z),4,4),complex)
 for i,w in enumerate(z):
  if abs(w.real)<=e*(1+1e-9) and abs(w.imag)<=e*(1+1e-9):k=0;rho=max(abs(w),e/2);sign=1
  elif abs(w.real-1)<=e*(1+1e-9) and abs(w.imag)<=e*(1+1e-9):k=1;rho=max(abs(w-1),e/2);sign=1
  else:k=2;rho=max(abs(w),e/2);sign=-1
  h[i]=(frames[k]*rho**(2*sign*mus[k]))@frames[k].T
 return h

def root_error(a,b):
 cost=abs(np.asarray(a)[:,None]-np.asarray(b)[None,:]);i,j=linear_sum_assignment(cost)
 return float(cost[i,j].max()),j

def holonomies(mesh,h):
 hx=mesh.op(mesh.dx,h);hy=mesh.op(mesh.dy,h)
 coef=np.linalg.solve(h,(hx-1j*hy)/2).reshape(mesh.nx,mesh.ny,4,4)
 interp=RegularGridInterpolator((mesh.x,mesh.y),coef,bounds_error=True)
 hb=RegularGridInterpolator((mesh.x,mesh.y),h.reshape(mesh.nx,mesh.ny,4,4))([.5,0])[0].real
 w,v=np.linalg.eigh(hb);sq=(v*np.sqrt(w))@v.T;si=(v*(1/np.sqrt(w)))@v.T
 def segment(p,q,initial):
  dz=q-p
  def rhs(t,m):
   z=p+t*dz;a=interp([z.real,z.imag])[0]
   return (-dz*a@m.reshape(4,4)).ravel()
  sol=solve_ivp(rhs,(0,1),initial.ravel(),rtol=1e-8,atol=1e-10,max_step=.08)
  if not sol.success:raise RuntimeError(sol.message)
  return sol.y[:,-1].reshape(4,4)
 out=[]
 for p in [0.,1.]:
  r=.3
  # Each CCW square loop has a real stem from the common real basepoint.
  if p==0: vertices=[.5,p+r,p+r+1j*r,p-r+1j*r,p-r-1j*r,p+r-1j*r,p+r,.5]
  else: vertices=[.5,p-r,p-r-1j*r,p+r-1j*r,p+r+1j*r,p-r+1j*r,p-r,.5]
  m=np.eye(4,dtype=complex)
  for aa,bb in zip(vertices[:-1],vertices[1:]):m=segment(aa,bb,m)
  out.append(sq@m@si)
 return out

def extract(us,data):
 qs=[];diagnostics=[]
 for u,phase in zip(us,[data['a'],data['b']]):
  rr=np.exp(2j*np.pi*np.array(phase));error,perm=root_error(np.linalg.eigvals(u),rr)
  # Real eigenframe extraction; phases assigned independently by nearest roots.
  m=(u.real+u.real.T)/2+.371*(u.imag+u.imag.T)/2
  _,q=np.linalg.eigh(m);diag=np.diag(q.T@u@q);_,perm=root_error(diag,rr)
  q=q[:,np.argsort(perm)]
  qs.append(q)
  diagnostics.append({'root_error':error,'unitarity_error':float(np.linalg.norm(adj(u)@u-np.eye(4))),'symmetry_error':float(np.linalg.norm(u-u.T)),'offdiag_error':float(np.linalg.norm(q.T@u@q-np.diag(np.diag(q.T@u@q))))})
 o=qs[0].T@qs[1]
 if np.linalg.det(o)<0:o[:,0]*=-1
 d=np.diag(np.exp(1j*np.pi*np.array(data['a'])));b=np.diag(np.exp(2j*np.pi*np.array(data['b'])))
 err,_=root_error(np.linalg.eigvals(d@o@b@o.T@d),np.exp(2j*np.pi*np.array(data['target'])))
 return o,diagnostics,err

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['fixture','control','target']);p.add_argument('--level',type=int,default=0);args=p.parse_args()
 if args.mode=='fixture':make_fixture();return
 start=time.perf_counter()
 if args.mode=='control':
  mesh=Mesh(np.linspace(-1,1,17),np.linspace(-1,1,17),[]);expected=manufactured(mesh.z).reshape(-1,4,4)
  h,logs,status=mesh.solve(expected)
  result={'status':status,'metric_error':float(np.max(abs(h-expected))),'nodes':len(h),'seconds':time.perf_counter()-start,'logs':logs}
 else:
  data=json.loads((ROOT/'fixture.json').read_text());mesh,e,r=target_mesh(args.level)
  h,logs,status=mesh.solve(boundary_metric(mesh,data,e))
  # Always extract if positive: wrong holonomy is the intended falsifier.
  us=holonomies(mesh,h);o,diag,err=extract(us,data)
  result={'status':status,'level':args.level,'epsilon':e,'outer_radius':r,'nodes':len(h),'unknowns':len(mesh.free)*16,'seconds':time.perf_counter()-start,'peripheral':diag,'target_root_error':err,'O':o.tolist(),'logs':logs}
  np.savez_compressed(ROOT/f'metric-{args.level}.npz',x=mesh.x,y=mesh.y,h=h)
 name='control' if args.mode=='control' else f'target-{args.level}'
 (ROOT/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 print('RESULT '+json.dumps({k:v for k,v in result.items() if k not in ['logs','O']}),flush=True)
if __name__=='__main__':main()
