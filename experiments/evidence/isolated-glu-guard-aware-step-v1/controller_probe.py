import importlib.util,json,hashlib
from pathlib import Path
p=Path('/private/tmp/modelforge-guard-aware-step-review');repo=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
spec=importlib.util.spec_from_file_location('ind_controller',repo/'examples/isolated_glu/guard_aware_controller.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
config=p/'synthetic-prereg.json';review=p/'synthetic-review.json';write(config,{});write(review,{})
m.verify_source=lambda *a:({},'synthetic-stable-head')
results={}
for mode in ['changed','nochange','leftover','mutated','failedroot','scopefailure']:
 root=p/('controller-'+mode)
 calls=[]
 def engine(command,request,**kw):
  calls.append(request)
  if len(calls)==1:return {'process_monitor_scope':'process_and_children','memory_samples':2}
  worker=Path(request['output']);step=worker/'step';step.mkdir(parents=True)
  changed=mode!='nochange' and mode!='leftover'
  write(step/'report.json',{'status':'failed' if mode=='failedroot' else 'completed','changed':changed,'accepted':[1] if changed else []})
  if changed or mode=='leftover':(step/'candidate.npz').write_bytes(b'syntheticcandidate')
  write(step/'artifact-manifest.json',{'status':'completed','files':{str(x.relative_to(step)):sha(x) for x in step.iterdir() if x.is_file()}})
  wr={'status':'completed','inputs_unchanged':True,'candidate_built':changed,'step_report_sha256':sha(step/'report.json'),'artifact_manifest_sha256':sha(step/'artifact-manifest.json')}
  if changed:wr['candidate_sha256']=sha(step/'candidate.npz')
  write(worker/'worker-report.json',wr)
  if mode=='mutated':(step/'uncommitted-extra').write_bytes(b'bad')
  return {'process_monitor_scope':'process_only_children_unavailable' if mode=='scopefailure' else 'process_and_children','memory_samples':2,'changed':changed,'worker_report_sha256':sha(worker/'worker-report.json')}
 try:r=m.run_protocol(repo,root,{},config,review,engine=engine)
 except m.ProtocolError:r=json.loads((root/'controller-report.json').read_text())
 expected=mode in ['changed','nochange']
 assert (r['status']=='completed')==expected
 assert r['evaluation_eligible']==(mode=='changed')
 assert len(calls)==2
 results[mode]={'status':r['status'],'evaluation_eligible':r['evaluation_eligible']}
results['actual_input_reads']=0;results['target_queries']=0;results['real_engine_calls']=0
(p/'controller-probe-results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))
