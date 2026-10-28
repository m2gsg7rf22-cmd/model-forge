"""Tiny scalar/mocked-controller controls; no registered factory/RNG/data."""
import sys,json,tempfile,hashlib
from pathlib import Path
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-fresh-protocol-review');sys.path[:0]=[str(R/'src'),str(R/'examples/isolated_glu')]
import fresh_rank_evaluation as e
import fresh_rank_controller as c
result={}
reference=e.reference_summary(np.diag([1.,0.]));b=np.array([[1.,1e-200]])
component=e.component_residuals(reference,b);relative=e.energy_loss(np.diag([1.,0.]),b)
result['representable_tiny_component']={'value':float(component[0]),'expected':1e-200,'relative_Frobenius':relative,'false_zero':bool(component[0]==0 and relative>0)}
reference=e.reference_summary(np.diag([1.,1e-200,0.]));b=np.array([[1.,0.,0.],[0.,1.,1e-200]])
try:e.component_residuals(reference,b)
except ValueError:result['underrepresentable_weighted_component_rejected']=True
else:result['underrepresentable_weighted_component_rejected']=False
# Isolate end-of-controller inventory check with all scientific stage validation mocked.
# The mock never launches a subprocess, factory, dataset, or reference callback.
root=Path(tempfile.mkdtemp(prefix='mock-inventory-',dir=P));repo=root/'repository';repo.mkdir();ex=root/'experiments';ex.mkdir();out=ex/'run';config=repo/'config.json';review=repo/'review.json';config.write_text('{}');review.write_text('{}');head='literal-source'
originals={k:getattr(c,k) for k in ['verify_source','public_boundary','validate_final','stage_report']};olddisk=c.shutil.disk_usage
c.verify_source=lambda *a:({},head);c.public_boundary=lambda *a:{};c.validate_final=lambda *a:None;c.shutil.disk_usage=lambda *a:type('D',(),{'free':3*1024**3})()
collection={'status':'completed','source_commit':head,'calls_reserved':545280,'calls_attempted':545280,'calls_returned':545280,'calls_persisted':545280,'callback_batches':768,'certificate':False,'private_gate':'closed','truth_inventory':{'shards':1608},'public_inventory':{'shards':144},'public_cases_sha256':'literal-cases','public_manifest_sha256':'literal-manifest'}
est={'status':'completed','source_commit':head,'seal_sha256':'literal-seal'};evaluation={'status':'completed','source_commit':head,'rule_passes':{'gap':False,'stable':False}}
c.stage_report=lambda output,stage,receipt,source: {'collection':collection,'estimator':est,'evaluation':evaluation}[stage]
calls=[]
def engine(command,payload,**kw):
 calls.append(payload)
 assert kw['memory_limit_mb']==512 and kw['timeout_seconds']<=180 and kw['output_limit_bytes']==2048
 i=len(calls)
 if i==3:
  d=out/'collection';d.mkdir();(d/'collection-report.json').write_text('{}')
  for name,field in [('truth-commitment.json','truth_commitment_sha256'),('seed-commitment.json','seed_commitment_sha256')]:
   (d/name).write_text('{}');collection[field]=c.sha(d/name)
 elif i==4:
  d=out/'estimator';d.mkdir();(d/'estimator-report.json').write_text('{}')
 elif i==5:
  d=out/'evaluation';d.mkdir();(d/'evaluation-report.json').write_text('{}');(out/'extra-uncommitted.npz').write_bytes(b'Not a committed archive')
 return {'status':'passed','process_monitor_scope':'process_and_children','memory_samples':1}
try:
 try:report=c.run_protocol(repo,out,config,review,engine=engine,clock=lambda:0.)
 except c.ProtocolError as exc:
  result['uncommitted_root_file_rejected_at_final_boundary']=len(calls)==5 and (out/'extra-uncommitted.npz').exists();result['mock_terminal_error']=str(exc);assert len(calls)==5, 'Mock failed before final boundary'
 else:result['uncommitted_root_file_rejected_at_final_boundary']=False;result['mock_final_status']=report['status']
finally:
 for k,v in originals.items():setattr(c,k,v)
 c.shutil.disk_usage=olddisk
result.update(source_sha256={n:hashlib.sha256((R/'examples/isolated_glu'/n).read_bytes()).hexdigest() for n in ['fresh_rank_evaluation.py','fresh_rank_controller.py']},mocked_stage_calls=len(calls),actual_subprocesses_launched=0,actual_factory_RNG_target_queries_fitting_data=0,full_protocol_approved=False)
(P/'current-boundary-probe-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
