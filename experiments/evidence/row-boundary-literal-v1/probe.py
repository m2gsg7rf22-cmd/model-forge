"""Literal algebraic counterexample only; no registered data, RNG or queries."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
r=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
sys.path.insert(0,str(r/'examples/isolated_glu'))
from rank_selection import select
source=r/'examples/isolated_glu/rank_selection.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='1382f141fd4adf7a80106df75ccedfba4b55b9dd2b877ba4399becc4646f870b'
delta=1e-6
narrow=np.array([[1.,0.,0.],[0.,delta,0.]])
wide=np.diag([1.,delta,delta])
items=[]
for name,a,expected in [('row_limited',narrow,2),('square_control',wide,1)]:
    result=select(a,a,a)
    ranks={k:v['rank'] for k,v in result['choices'].items()}
    assert ranks['gap']==ranks['stable']==expected
    assert result['certificate'] is False
    assert all(not c['certificate'] for c in result['choices'].values())
    floor=float(result['diagnostics']['resolution_floor'])
    if name=='row_limited':
        assert delta**2>floor
        assert result['spectra']['richardson'].tolist()==[1.,delta,0.]
        assert result['diagnostics']['gap_ratios'][2,1]>result['diagnostics']['gap_ratios'][2,0]
    items.append(dict(name=name,shape=list(a.shape),input_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        ranks=ranks,spectrum=result['spectra']['richardson'].tolist(),resolution_floor=floor,
        gap_ratios=result['diagnostics']['gap_ratios'][2].tolist(),
        stability=result['diagnostics']['stability'].tolist(),certificate=False))
output=dict(status='completed_literal_counterexample',delta=delta,items=items,
    frozen_selection_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),new_queries=0,
    new_registered_data=0,RNG=0,teacher_forward=0,fit=0,scientific_fresh_validation=False,
    proves='Structural padding can dominate the frozen spectral gap even with identical-view stability.',
    does_not_prove='Adding a probe fixes any SwiGLU failure or identifies true rank/width under unknown noise.',private_gate='closed')
p=Path(__file__).with_name('results.json');p.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
time.sleep(.15)
print(json.dumps(dict(status='completed',results_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),new_queries=0)))
