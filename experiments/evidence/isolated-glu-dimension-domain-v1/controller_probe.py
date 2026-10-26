"""Mocked protocol flow/schema only; no registered worker, RNG or global marker."""
import sys,importlib.util,json
from pathlib import Path
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-domain-run-review');sys.path.insert(0,str(R/'examples/isolated_glu'));sp=importlib.util.spec_from_file_location('ind_domain_controller',R/'examples/isolated_glu/scaling_domain_controller.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
origsha=m.sha;origread=m.read_json;origiter=Path.iterdir;original_validate=m.validate_worker
root=P/'virtual-worker-schema';worker=root/'worker';arrays=worker/'arrays';arrays.mkdir(parents=True);repo=P/'virtualrepo';repo.mkdir();marker=repo.parent/'experiments/isolated-glu-dimension-domain-v1-attempt.json';assert not marker.exists();head='synthetic-not-actual';cfg='0'*64
fullnames=m.expected_names();assert len(fullnames)==4704;inv={'shards':4704,'bytes_reserved':100,'largest_shard_bytes':10};markerdata={'policy':m.POLICY,'calls_reserved':1454080,'source_commit':head,'config_sha256':cfg,'output':str(worker.absolute()),'automatic_retry':False};markerhash='a'*64;manifesthash='b'*64
m.sha=lambda p:markerhash if p==marker else manifesthash if p==arrays/'manifest.jsonl' else origsha(p)
m.read_json=lambda p:markerdata if p==marker else origread(p)
m.verify_manifest=lambda p:dict(inv)
valid={'status':'completed','source_commit':head,'cases':[{'synthetic':i} for i in range(16)],'marker':str(marker),'marker_sha256':markerhash,'artifact_inventory':inv,'artifact_manifest_sha256':manifesthash,'inventory_plan':m.inventory_plan(),**{k:1454080 for k in ['calls_reserved','calls_attempted','calls_returned','calls_persisted']}}
results={}
for mode in ['success','wrong_counts','marker_source','missing_artifact_name']:
 report=json.loads(json.dumps(valid));names=set(fullnames)
 if mode=='wrong_counts':report['calls_returned']-=1
 if mode=='marker_source':markerdata['source_commit']='changed'
 else:markerdata['source_commit']=head
 if mode=='missing_artifact_name':names.remove(next(iter(names)))
 (worker/'worker-report.json').write_text(json.dumps(report));summary={'worker_report_sha256':origsha(worker/'worker-report.json'),**{k:1454080 for k in ['calls_reserved','calls_attempted','calls_returned','calls_persisted']}}
 Path.iterdir=lambda p:iter([arrays/name for name in names|{'manifest.jsonl'}]) if p==arrays else origiter(p)
 try:
  try:m.validate_worker(root,summary,head,cfg,repo)
  except ValueError:passed=False
  else:passed=True
 finally:Path.iterdir=origiter
 assert passed==(mode=='success');results['schema_'+mode]=passed
assert not marker.exists()
# Actual run_protocol with mock engine/source gates/validated synthetic report only.
m.sha=origsha;m.read_json=origread;m.validate_worker=lambda *a:{'synthetic_only':True}
config=P/'synthetic-config.json';review=P/'synthetic-review.json';config.write_text('{}');review.write_text('{}')
for mode in ['success','scope_failure','transport_failure','worker_tree_failure','post_source_change']:
 calls=[];sourcecalls=[]
 def source(*args):
  sourcecalls.append(1);return {'synthetic_only':True},'changed-head' if mode=='post_source_change' and len(sourcecalls)>1 else 'fixed-synthetic-head'
 m.verify_source=source
 def engine(command,payload,**kw):
  calls.append(payload);i=len(calls)
  assert kw['memory_limit_mb']==512 and kw['output_limit_bytes']==2048 and 0<kw['timeout_seconds']<=180
  scope='process_only_children_unavailable' if (mode=='scope_failure' and i==1) or (mode=='worker_tree_failure' and i==3) else 'process_and_children'
  return {'status':'failed' if mode=='transport_failure' and i==2 else 'passed','process_monitor_scope':scope,'memory_samples':2}
 output=P/('mocked-flow-'+mode)
 try:out=m.run_protocol(R,output,config,review,engine=engine,clock=lambda:0.)
 except m.ProtocolError:out=json.loads((output/'controller-report.json').read_text())
 assert (out['status']=='completed')==(mode=='success') and out['diagnostic_completed']==(mode=='success')
 if mode=='scope_failure':assert len(calls)==1 and not out['worker_started'] and out['scope-engine_receipt']['process_monitor_scope']=='process_only_children_unavailable'
 if mode=='transport_failure':assert len(calls)==2 and not out['worker_started']
 if mode=='worker_tree_failure':assert out['worker-engine_receipt']['process_monitor_scope']=='process_only_children_unavailable'
 results['flow_'+mode]={'status':out['status'],'engine_calls':len(calls),'diagnostic_completed':out['diagnostic_completed']}
results.update(actual_registered_contexts_called=0,actual_worker_processes=0,registered_RNG_calls=0,scientific_global_markers_created=0,virtual_inventory_and_marker_used_for_schema=True)
(P/'controller-probe-results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))
