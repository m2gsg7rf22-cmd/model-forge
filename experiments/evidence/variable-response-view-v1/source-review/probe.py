import sys,json,hashlib
from pathlib import Path
from unittest.mock import patch
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
sys.path[:0]=[str(R/'examples/isolated_glu'),str(R/'src')]
import variable_response_controller as c
import variable_response_protocol as p
from variable_response_inventory import inventory
O=Path('/private/tmp/modelforge-variable-response-protocol-review');T=O/'literal';T.mkdir(exist_ok=True)
checks={}
# Full logical chain with literal hashes/shapes, no arrays or registered source reads.
chain=p.DeliveryChain(T/'max-chain.jsonl')
for i in range(204):chain.append(f'job{i//34:02d}',i%34,(2112,32),(2112,32),'1'*64,'2'*64)
r=chain.verify();assert r['pairs']==204 and r['bytes']<=inventory()['delivery_upper_bound']
checks['204_literal_ordered_records_fit_declared_chain_bound']=True
# Same-length external mutation invalidates verification; it is not silently accepted.
old=(T/'max-chain.jsonl').read_bytes();(T/'max-chain.jsonl').write_bytes(old.replace(b'1111',b'3333',1))
try:chain.verify()
except ValueError:pass
else:raise AssertionError('chain accepted modified digest')
checks['same_length_chain_mutation_rejected']=True
# Explicit ordinary append failure terminates this object; no invented count advance.
short=p.DeliveryChain(T/'short-chain.jsonl'); realopen=Path.open
class Writer:
 def __init__(self,f):self.f=f
 def __enter__(self):return self
 def __exit__(self,*args):self.f.close()
 def write(self,b):self.f.write(b[:3]);return 3
 def flush(self):self.f.flush()
 def fileno(self):return self.f.fileno()
def partialopen(path,*args,**kwargs):
 f=realopen(path,*args,**kwargs)
 return Writer(f) if path==short.path and args and args[0]=='ab' else f
with patch.object(Path,'open',partialopen):
 try:short.append('job00',0,(4,1),(4,1),'1'*64,'2'*64)
 except OSError:pass
 else:raise AssertionError('short append accepted')
assert short.failed and short.count==0 and short.path.stat().st_size==3
try:short.append('job00',0,(4,1),(4,1),'1'*64,'2'*64)
except ValueError:pass
else:raise AssertionError('append retry accepted')
checks['short_chain_write_is_terminal_with_uncommitted_prefix_preserved']=True
# Invalid preflight is recorded and prevents even metadata loader/marker.
repo=T/'repository';repo.mkdir();cfg=repo/'config.json';rev=repo/'review.json';p.save(cfg,{});p.save(rev,{})
output=T/'experiments'/'preflight';seen=[]
bad=dict(status='passed',process_monitor_scope='process_only_children_unavailable',memory_samples=2,peak_sampled_rss_bytes=100,memory_limit_mb=512)
def fake_engine(*args,**kwargs):seen.append(1);return bad
def forbidden(*args):raise AssertionError('metadata/data opened after failed preflight')
with patch.object(c,'verify_source',lambda *args:({},'literal-head')):
 try:c.run_protocol(repo,output,cfg,rev,engine=fake_engine,_manifest_loader=forbidden,_view_factory=forbidden)
 except ValueError:pass
 else:raise AssertionError('bad scope accepted')
r=p.read(output/'controller-report.json');assert r['status']=='failed' and r['scope_receipt']==bad and not r['marker_consumed']
assert len(seen)==1 and not (T/'experiments'/p.MARKER).exists()
checks['failed_scope_receipt_saved_before_refusal_without_metadata_or_marker']=True
# Outer engine exception: preserve unknown receipt/counters, no false zero resource.
outer=T/'experiments'/'outer-error'
req=dict(repo=str(repo),output=str(outer),config=str(cfg),review=str(rev))
def explode(*args,**kwargs):outer.mkdir();raise RuntimeError('literal engine exception no receipt')
try:c.launch(req,engine=explode)
except RuntimeError:pass
else:raise AssertionError('engine exception accepted')
r=p.read(outer/'outer-monitor.json');assert r['status']=='failed' and r['resource_receipt_unavailable'] is True and r['controller_counters_unknown'] is True and 'receipt' not in r
checks['engine_exception_preserves_explicit_unknown_resource_and_counters']=True
# Completed child + process-only outer may not become successful.
outer2=T/'experiments'/'outer-scope';req2=dict(req,output=str(outer2))
def wrongtree(*args,**kwargs):
 outer2.mkdir();p.save(outer2/'controller-report.json',{'status':'completed'})
 return dict(bad,status='completed')
try:c.launch(req2,engine=wrongtree)
except ValueError:pass
else:raise AssertionError('process-only outer accepted')
r=p.read(outer2/'outer-monitor.json');assert r['status']=='failed' and r['receipt']['status']=='completed' and r['resource_receipt_unavailable'] is False
checks['completed_child_cannot_override_failed_outer_tree_and_receipt_retained']=True
# Descriptor bound arithmetic at fixed production maximum, descriptors only.
pairs=[]
for job in range(6):
 for step in range(2):
  for probe in range(17):
   base=f'c{job//2:02d}-'+('fp64' if job%2==0 else 'fp32')
   a=lambda field:dict(name=f'{base}-s{step}-x{probe:02d}-{field}.npz',field=field,shape=[2112,32],digest='a'*64)
   pairs.append(dict(job_id=f'job{job:02d}',index=step*17+probe,inputs=a('inputs'),responses=a('responses')))
seal=dict(pairs=pairs,public_metadata={x:'a'*64 for x in ['audit','collection','progress','commitments']})
assert len(p.canonical(seal))<=inventory()['descriptor_seal_upper_bound']
checks['204_scalar_descriptors_fit_exact_seal_bound_without_arrays']=True
# Current prereg has exact runtime/dependency/inventory values, no source gate bypass execution.
config=p.read(R/'experiments/preregistrations/variable-response-view-v1.json')
assert p.canonical(config)==p.canonical(c.config_template(R))
checks['prereg_exactly_matches_runtime_source_pins_policy_inventory']=True
result=dict(passed=True,checks=checks,scope='small literal/fake engine/scalar metadata controls only; no actual transcript/RNG/teacher/query/fit',actual_transcript_reads=0,actual_resource_proof=False,maximum_chain_bytes=chain.bytes,maximum_descriptor_seal_bytes=len(p.canonical(seal)),runtime_sha256={n:p.digest(R/n,2*1024**2) for n in sorted(c.DEPENDENCIES)})
(O/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'checks':len(checks),'actual_transcript_reads':0}))
