import importlib.util,json,hashlib
from pathlib import Path
import numpy as np
p=Path('/private/tmp/modelforge-guard-aware-step-review');repo=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
spec=importlib.util.spec_from_file_location('ind_worker_final',repo/'examples/isolated_glu/guard_aware_worker.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
result={}
for mode in ['stable','mutated','missing']:
 inp=p/('synthetic-finally-input-'+mode);inp.write_bytes(b'originalsynthetic')
 digest=hashlib.sha256(inp.read_bytes()).hexdigest();root=p/('worker-finally-'+mode)
 def loader(paths,config,sink):
  sink({'synthetic':digest})
  assert json.loads((root/'worker-report.json').read_text())['inputs_before']=={'synthetic':digest}
  a=np.ones((1,1));return {'synthetic':digest},{'gates':a,'couplings':a},{k:a for k in ['train_x','train_y','guard_x','guard_y']}
 def fail(*args,**kw):
  if mode=='mutated':inp.write_bytes(b'changed')
  if mode=='missing':inp.unlink()
  raise RuntimeError('original synthetic failure')
 m.read_data=loader;m.step.run_step=fail
 try:m.build({'synthetic':inp},root,{}, {'process_monitor_scope':'process_and_children','memory_samples':2})
 except RuntimeError as e:assert str(e)=='original synthetic failure'
 else:raise AssertionError('must fail')
 r=json.loads((root/'worker-report.json').read_text());assert r['status']=='failed' and not r['candidate_built'] and r['error']=='original synthetic failure'
 expected={'stable':True,'mutated':False,'missing':None}[mode]
 assert r['inputs_unchanged'] is expected
 assert ('input_check_error' in r)==(mode!='stable')
 result[mode]={'inputs_unchanged':r['inputs_unchanged'],'original_exception_preserved':True,'before_persisted_before_loader_return':True}
result.update(actual_input_reads=0,actual_fits=0,target_queries=0)
(p/'worker-finally-probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
