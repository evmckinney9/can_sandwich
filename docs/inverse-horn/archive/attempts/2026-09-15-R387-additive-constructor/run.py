import json,time
from pathlib import Path
from constructor import solve
p=Path(__file__).parent
out=[]
for case in json.loads((p/'inputs.json').read_text()):
    for blocks in (False,True):
        start=time.perf_counter()
        q,info=solve(case['alpha'],case['beta'],case['gamma'],blocks=blocks)
        row=dict(name=case['name'],blocks=blocks,seconds=time.perf_counter()-start,Q=None if q is None else q.tolist(),**info)
        out.append(row)
        print(case['name'],blocks,info,flush=True)
(p/'results.json').write_text(json.dumps(out,indent=2)+'\n')
