import sys,importlib.util,json
from pathlib import Path
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-domain-run-review');sys.path.insert(0,str(R/'examples/isolated_glu'));sp=importlib.util.spec_from_file_location('ind_run_draft',R/'examples/isolated_glu/scaling_domain_run.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=m.ShardStore(P/'fixed-output-copy');ledger=m.QueryLedger(P/'fixed-output-progress.json',4);returned=np.broadcast_to(np.array([1.]),(1_000_000_000,32));orig=m.np.array;seen=[]
def guard(a,*args,**kw):
 if a is returned:seen.append(True);raise RuntimeError('oversized response copied before guard')
 return orig(a,*args,**kw)
m.np.array=guard
try:
 try:ledger.query(s,'test',np.ones((4,2)),lambda x:returned,2)
 except ValueError:pass
 else:raise AssertionError('oversized response accepted')
finally:m.np.array=orig
assert not seen and ledger.counts['calls_attempted']==4 and ledger.counts['calls_returned']==4 and ledger.counts['calls_persisted']==0
reference=np.ones(10);reference[-1]=0;estimate=reference.copy();estimate[-1]=np.nextafter(0.,1.);relative=m._relative(estimate,reference);assert relative==0
r={'oversized_return_view_now_rejected_before_copy':True,'nonzero_relative_ratio_underflow_returns_zero':True,'literal_tiny_difference':float(estimate[-1]),'registered_seeds_used':0,'reference_queries':0,'rng_calls':0}
(P/'fixed-schema-probe-results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
