import importlib.util,json,hashlib
from pathlib import Path
P=Path('/private/tmp/modelforge-guard-aware-evaluation-review');R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');sp=importlib.util.spec_from_file_location('final_eval_controller',R/'examples/isolated_glu/guard_aware_eval_controller.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
original_sha=m.sha;original_read=m.read_json
fake_fit=P/'virtual-fit-parent';marker=fake_fit/'public941033-evaluation-attempt.json'
assert not marker.exists()
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
results={}
for mode in ['quality_pass','quality_fail','resource_fail','input_mutation','wrong_counts','marker_tamper']:
 inputs={k:hashlib.sha256(k.encode()).hexdigest() for k in ['candidate','baseline','fit_controller','fit_worker','fit_step','fit_artifact_manifest','outcome_audit']};config={'inputs':inputs};canonical=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();virtual={'evaluation_seed':941033,'calls_reserved':2048,'candidate_sha256':inputs['candidate'],'baseline_sha256':inputs['baseline'],'config_content_sha256':canonical};markerhash=hashlib.sha256(json.dumps(virtual).encode()).hexdigest()
 m.sha=lambda path:markerhash if path==marker else original_sha(path)
 m.read_json=lambda path:dict(virtual,evaluation_seed=0) if path==marker and mode=='marker_tamper' else virtual if path==marker else original_read(path)
 m.verify_source=lambda *args:(config,'synthetic-clean-head')
 cfg=P/f'controller-config-{mode}.json';rev=P/f'controller-review-{mode}.json';write(cfg,config);write(rev,{})
 root=P/f'final-controller-{mode}';calls=[]
 def engine(command,request,**kw):
  calls.append(1)
  if len(calls)==1:return {'process_monitor_scope':'process_and_children','memory_samples':2}
  worker=Path(request['output']);batches=worker/'batches';batches.mkdir(parents=True);(batches/'paired-validation.npz').write_bytes(b'no-real-reference-synthetic-commitment')
  counters={k:2048 for k in ['calls_reserved','calls_attempted','calls_returned','calls_persisted']};quality=mode=='quality_pass'
  pair={'status':'completed','seed':941033,'groups':[{'scale':s} for s in [.01,1.,10.,100.]],'development_max_group_relative':.001,'all_groups_relative_at_most_1e_minus3':quality,'validation_sha256':original_sha(batches/'paired-validation.npz'),**counters};write(batches/'progress.json',pair)
  wr={'status':'completed','inputs_unchanged':True,'inputs_before':inputs.copy(),'inputs_after':inputs.copy(),'evaluation':pair,'paired_report_sha256':original_sha(batches/'progress.json'),'candidate_sha256':inputs['candidate'],'baseline_sha256':inputs['baseline'],'persistent_marker':str(marker),'persistent_marker_sha256':markerhash,**counters}
  if mode=='input_mutation':wr['inputs_after']['candidate']='changed'
  if mode=='wrong_counts':wr['calls_attempted']=1536
  write(worker/'worker-report.json',wr)
  return {'worker_report_sha256':original_sha(worker/'worker-report.json'),'criterion_met':quality,'process_monitor_scope':'process_only_children_unavailable' if mode=='resource_fail' else 'process_and_children','memory_samples':2,**counters}
 try:res=m.run_protocol(R,root,{'fit_controller':fake_fit/'controller-report.json'},cfg,rev,engine=engine)
 except m.ProtocolError:res=json.loads((root/'controller-report.json').read_text())
 success=mode in ['quality_pass','quality_fail'];assert (res['status']=='completed')==success;assert res['evaluation_completed']==success;assert res['quality_criterion_met']==(mode=='quality_pass' if success else None)
 results[mode]={'status':res['status'],'quality_criterion_met':res['quality_criterion_met']}
assert not marker.exists()
results.update(actual_marker_created=False,actual_reference_factory_calls=0,actual_seed_points_generated=False,source_gate_replaced_for_synthetic_flow_only=True,marker_contents_in_memory_only=True)
(P/'final-controller-probe-results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))
