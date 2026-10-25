import importlib.util,json
from pathlib import Path
P=Path('/private/tmp/modelforge-guard-aware-evaluation-review');R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');spec=importlib.util.spec_from_file_location('ind_marker',R/'examples/isolated_glu/guard_aware_evaluate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root=P/'synthetic-marker-fit';root.mkdir()
p=m.reserve_once(root,{'calls_reserved':8,'output':'synthetic'},seed=29)
try:m.reserve_once(root,{},seed=29)
except FileExistsError:pass
else:raise AssertionError('repeat reservation allowed')
try:m.reserve_once(root,{'unserializable':float('nan')},seed=30)
except ValueError:pass
else:raise AssertionError('NaN marker unexpectedly serialized')
assert (root/'public30-evaluation-attempt.json').exists()
try:m.reserve_once(root,{},seed=30)
except FileExistsError:pass
else:raise AssertionError('failed-write reservation retried')
r={'generic_seeds':[29,30],'same_fit_new_output_still_refused':True,'failed_marker_write_still_blocks_retry':True,'partial_marker_does_not_prove_completed_reservation_metadata':True,'actual_seed_marker_created':False,'actual_reference_factory_calls':0}
(P/'marker-probe-results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
