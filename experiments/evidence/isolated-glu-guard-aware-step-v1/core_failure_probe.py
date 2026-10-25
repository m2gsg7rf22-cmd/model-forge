import runpy,json,hashlib
from pathlib import Path
import numpy as np
p=Path('/private/tmp/modelforge-guard-aware-step-review')
z=runpy.run_path(str(p/'core_probe.py'));m=z['m']
g=np.ones((1,1));c=np.ones((1,1,1));x=np.linspace(.2,1.6,8)[:,None];gx=x*.9
y=m.gradient.forward(g*1.04,c,x);gy=m.gradient.forward(g*1.00005,c,gx)
root=p/'independent-core-rejected-cache'
original=m.cached_trial
m.cached_trial=lambda *a:original(*a)+1
try:m.run_step(root,g,c,x,y,gx,gy,rows=2)
except m.StepError:pass
else:raise AssertionError('cache failure did not stop')
r=json.loads((root/'report.json').read_text());manifest=json.loads((root/'artifact-manifest.json').read_text())
assert r['status']=='failed' and not r['accepted'] and manifest['status']=='failed'
with np.load(root/'trials/0-predictions.npz') as a:
 assert np.max(np.abs(a['full_train_prediction']-a['cached_train_prediction']))>.99
with np.load(root/'derived-initial.npz') as a:
 np.testing.assert_array_equal(a['weighted_target'],y*a['weights'][:,None])
for name,digest in manifest['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
# These physical guard points reject even the smallest unchanged direction trial.
m.cached_trial=original
root2=p/'independent-core-eight-reject'
candidate,report=m.run_step(root2,g,c,x,y,gx,gy,rows=2)
assert candidate is None and report['status']=='completed' and len(report['trials'])==8 and not report['accepted']
assert all(v['training_decreased'] and not v['guard_passed'] for v in report['trials'])
result={'synthetic_only':True,'cache_failure_raw_predictions_preserved':True,'failure_inventory_all_hashes_verified':True,'derived_initial_weighted_once_verified':True,'all8_nonlinear_guard_rejections':True,'no_candidate_on_nochange':True,'actual_data_reads':0,'target_queries':0}
(p/'core-failure-probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
