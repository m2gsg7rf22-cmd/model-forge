from pathlib import Path
import json,sys,hashlib
import numpy as np
from modelforge.execution import run_json
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-guard-aware-trace-review');S=R/'examples/isolated_glu/guard_step_evidence.py';W=O/'independent-worst72'
code=f"import importlib.util,json;from pathlib import Path;s=importlib.util.spec_from_file_location('store',{str(S)!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m);print(json.dumps(m.synthetic_transport(Path('independent-worst72'))))"
receipt=run_json([sys.executable,'-c',code],{},cwd=O,memory_limit_mb=512,timeout_seconds=60,output_limit_bytes=2048)
(O/'worstsize-resource-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert receipt['proposal_count']==72 and receipt['subset_count']==receipt['subset_linear_solves_attempted']==18432 and receipt['trial_count']==64 and receipt['status']=='completed';assert len(json.dumps(receipt).encode())<2048
manifest=json.loads((W/'manifest.json').read_text());assert manifest['closed'] and not manifest['failed'] and len(manifest['proposals'])==72;raw=0;totalbytes=0
for i,entry in enumerate(manifest['proposals']):
 p=W/'proposals'/entry['proposal_id'];assert hashlib.sha256((p/'trace.json').read_bytes()).hexdigest()==entry['trace_sha256'];assert hashlib.sha256((p/'problem.npz').read_bytes()).hexdigest()==entry['problem_sha256'];assert hashlib.sha256((p/'events.jsonl').read_bytes()).hexdigest()==entry['journal_sha256'];t=json.loads((p/'trace.json').read_text());assert len(t['subsets'])==256 and t['result']['selected_mask']==255
 kn=np.array(t['normalization']['scaled_gram']);dn=np.array(t['normalization']['scaled_gradient']);bn=np.array(t['normalization']['normalized_constraints']);Z=np.array([v['solution'] for v in t['subsets']]);MU=np.array([v['normalized_duals'] for v in t['subsets']]);assert np.allclose(2*Z@kn.T+dn+MU@bn,0,atol=1e-12)
 for v,z in zip(t['subsets'],Z):assert np.allclose(bn[v['active']]@z,0,atol=1e-12)
 expected=-dn/(2*np.diag(kn));expected[:8]=0;assert np.allclose(t['result']['unclipped'],expected,atol=1e-12);raw+=len(Z);totalbytes+=entry['trace_bytes']
assert raw==18432 and totalbytes>2_000_000
result={'all72x256_raw_solutions_verified_without_new_solves':True,'analytic_diagonal_boundary_solutions_verified':True,'raw_solutions':raw,'trial_rows':64,'total_trace_bytes':totalbytes,'compact_monitored_receipt_bytes':len(json.dumps(receipt).encode()),'monitor_scope':receipt.get('process_monitor_scope'),'peak_sampled_rss_bytes':receipt.get('peak_sampled_rss_bytes'),'memory_samples':receipt.get('memory_samples'),'tree_monitor_scope_observed':receipt.get('process_monitor_scope')=='process_and_children','continuous_tree_memory_bound_claimed':False,'actual_candidate_fit':False,'actual_target_queries':0}
(O/'worstsize-probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
