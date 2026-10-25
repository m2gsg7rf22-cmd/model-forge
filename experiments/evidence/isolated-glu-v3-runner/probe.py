from pathlib import Path
import importlib.util, json, hashlib
import numpy as np
repo=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
spec=importlib.util.spec_from_file_location('v3',repo/'examples/isolated_glu/sealed_v3.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p=Path('/private/tmp/modelforge-v3-independent-review/synthetic'); p.mkdir(exist_ok=False); (p/'rescue').mkdir()
rng=np.random.default_rng(231)
d=rng.normal(size=(64,32)); d/=np.linalg.norm(d,axis=1,keepdims=True)
base={'directions':d[:63],'all_directions':np.tile(d[:63],(5,1))[:256],'residual':np.zeros(256)}
np.savez(p/'base-refined-solution.npz',**base)
m.write(p/'base-refine.json',{'directions_retained':63})
(p/'hessians.npz').write_bytes(b'public synthetic hash placeholder: not read by selection audit')
rescue={'directions':d,'starts':np.zeros((128,32)),'all_directions':np.zeros((128,32)),'residual':np.full(128,-1.0)}
np.savez(p/'rescue/coverage-solution.npz',**rescue)
np.savez(p/'refined-solution.npz',directions=d,all_directions=np.concatenate([base['all_directions'],rescue['all_directions']]),residual=np.concatenate([base['residual'],rescue['residual']]))
report={'branch':'rescue63','input_count':63,'base_solution_sha256':m.digest(p/'base-refined-solution.npz'),'base_refine_sha256':m.digest(p/'base-refine.json'),'heldout_read':False,'new_queries':0,'rescue_monitor':{'process_monitor_scope':'process_and_children','memory_samples':1,'peak_sampled_rss_bytes':1024,'memory_limit_mb':512}}
m.write(p/'direction-selection.json',report)
m.write(p/'rescue/report.json',{'source_sha256':m.registration()['solver_source_sha256']['examples/isolated_glu/coverage.py'],'input_sha256':{'hessians.npz':m.digest(p/'hessians.npz'),'refined-solution.npz':report['base_solution_sha256']},'retained_count':64,'input_count':63,'starts':128,'heldout_read':False,'new_queries':0})
m.write(p/'refine.json',{'directions_retained':64,'base_report_sha256':report['base_refine_sha256'],'rescue_report_sha256':m.digest(p/'rescue/report.json')})
print(json.dumps({'selection_audit_result':m.selection_audit(p),'retained_64th_abs_cosine_max_to_rescue_results':float(np.abs(rescue['all_directions']@d[63]).max()),'negative_loss':True,'zero_starts':True,'note':'Selection-only synthetic boundary probe; no actual H fitting or private data.'}))
