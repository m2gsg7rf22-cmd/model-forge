from pathlib import Path
from unittest.mock import patch
import sys,json
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-joint-response-full-source-review');sys.path[:0]=[str(R/'examples/isolated_glu'),str(R/'src')]
import joint_response_protocol as p
import joint_response_controller as c
from variable_response_protocol import save,read
T=O/'literal-control';T.mkdir(exist_ok=True);repo=T/'repository';repo.mkdir();review=repo/'review.json';save(review,{'private_gate':'closed','certificate':False})
try:p.verify_source(repo,repo/'MUST-NOT-READ-CONFIG',review)
except ValueError as e:assert 'approval' in str(e)
else:raise AssertionError('missing approval did not stop source gate')
checks={'missing_full_approval_rejects_before_config_or_source_or_data':True}
out=T/'experiments'/'outer-bad-tree';req=dict(repo=str(repo),output=str(out),config='literal-unused',review=str(review));seen=[]
def fake(*args,**kwargs):
 seen.append(kwargs);assert kwargs['memory_limit_mb']==512 and kwargs['timeout_seconds']==180 and kwargs['output_limit_bytes']==1024
 assert all(kwargs['env'][k]=='1' for k in p.THREAD_ENV)
 out.mkdir(parents=True);save(out/'controller-report.json',{'status':'completed'})
 return {'status':'completed','process_monitor_scope':'process_only','memory_samples':1,'peak_sampled_rss_bytes':10,'memory_limit_mb':512}
try:c.launch(req,engine=fake)
except ValueError:pass
else:raise AssertionError('bad outer scope accepted')
v=read(out/'outer-monitor.json');assert v['status']=='failed' and v['receipt']['status']=='completed' and v['resource_receipt_unavailable'] is False
checks['completed_child_bad_outer_scope_retained_failed_with_bounds_and_threads']=True
out2=T/'experiments'/'outer-engine-exception';req['output']=str(out2)
def raises(*a,**k):out2.mkdir(parents=True);raise OSError('literal engine no receipt')
try:c.launch(req,engine=raises)
except OSError:pass
else:raise AssertionError('engine exception accepted')
v=read(out2/'outer-monitor.json');assert v['status']=='failed' and v['resource_receipt_unavailable'] is True and v['controller_counters_unknown'] is True and 'receipt' not in v
checks['engine_exception_preserves_unknown_resources_and_counters']=True
(O/'small-control-results.json').write_text(json.dumps({'passed':True,'checks':checks,'scope':'small JSON/fake engine only; no realPCG/data/fit/largepreflight'},indent=2)+'\n');print(json.dumps({'passed':True,'checks':len(checks)}))
