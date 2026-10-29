from pathlib import Path
import importlib.util,json,sys
O=Path('/private/tmp/modelforge-joint-audit-case-review');p=O/'physical-chain-probe.py';spec=importlib.util.spec_from_file_location('literal_physical_fixture',p);f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
a=f.a
r=a.audit_case(**f.fixture())
assert r['status']=='disagreed' and r['physical_parameters_agreed'] is True and r['physical_predictions_agreed'] is True
assert r['normalized_predictions_agreed'] is False and r['normalized_physical_forward_agreed'] is False and r['all_predictions_agreed'] is False and r['selection_certified'] is False
assert r['prediction_rows_checked']==r['prediction_rows_visited']==8
for bad in [True,None,1e-15]:
 kw=f.fixture();kw['metadata']['sx']*=2;kw['metadata']['sy']=bad
 def forbidden(step):raise AssertionError('checkpoint called before scalar schema gate')
 kw['checkpoint']=forbidden
 try:a.audit_case(**kw)
 except ValueError:pass
 else:raise AssertionError('invalid sy bypassed after sx mismatch')
import joint_response_audit_inventory as inv
v=inv.inventory();assert v['prior_known_bytes']==857667407+1209==857668616
assert v['combined_artifacts_upper_bound']==sum(v[k] for k in ['prior_known_bytes','older_marker_reserve','root_metadata_bytes','case_metadata_bytes','temporary_metadata_bytes','audit_marker_and_temp_bytes','transport_encoded_upper_bound','transport_partial_reserve'])==868719624<1024**3
assert v['memory_planning_bytes']==sum(inv.memory_plan().values())==357023744 and v['preflight_literal_bytes']==256360448 and v['resource_proof'] is False
result={'passed':True,'checks':['wrong physical export RED now disagrees via independent normalizedselected prediction despite coefficient tolerance','reported sx mismatch cannot bypass sy bool/null/degenerate validation before checkpoint callback','prediction checked/visited counters8 on complete literal comparison','scalar total868719624 and memory357023744/preflight256360448 planning only'],'fixed_counterexample_status':r['status'],'actual_data_reads':0,'rng_calls':0,'fits':0};(O/'green-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'checks':len(result['checks'])}))
