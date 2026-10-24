import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import hashlib,json,time,resource,sys
from pathlib import Path
import numpy as np
start=time.monotonic()
base=Path('/Users/adamfaruhi123/.local/share/modelforge-research')
public=base/'experiments/isolated-glu-fullmap-v2-20261024-ec2623a/public/0'
result=base/'experiments/isolated-glu-coverage-development-v1-20261024/result'
repo=base/'repository'
report=json.loads((result/'report.json').read_text())
monitor=json.loads((result.parent/'monitored-report.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with np.load(public/'hessians.npz',allow_pickle=False) as z:H=z['hessian'].copy()
with np.load(public/'refined-solution.npz',allow_pickle=False) as z:known=z['directions'].copy()
with np.load(result/'coverage-solution.npz',allow_pickle=False) as z:D=z['directions'].copy()
i,j=np.triu_indices(32);weight=np.where(i==j,1.,np.sqrt(2.))
A=(H[...,i,j]*weight).reshape(-1,528)
_,s,Vt=np.linalg.svd(A,full_matrices=False);V=Vt[:128].T
# Lift the subspace to full Frobenius matrix coordinates; independently
# audit tangent geometry there, without calling parent's pack/tangent code.
matrices=np.zeros((128,32,32));matrices[:,i,j]=V.T/weight;matrices[:,j,i]=matrices[:,i,j]
Q=matrices.reshape(128,1024).T
loss=[];small=[];isometry=[];components=[]
for x in D:
 a=np.outer(x,x).ravel();r=a-Q@(Q.T@a);loss.append(float(r@r))
 L=np.stack([(np.outer(x,e)+np.outer(e,x)).ravel() for e in np.eye(32)],axis=1)
 T=np.linalg.qr(L)[0];isometry.append(float(np.linalg.norm(T.T@T-np.eye(32))))
 _,ss,ww=np.linalg.svd(T-Q@(Q.T@T),full_matrices=False)
 small.append(ss[-3:][::-1].tolist())
 components.extend([Q.T@(T@ww[-1]),Q.T@(T@ww[-2])])
cs=np.linalg.svd(np.stack(components),compute_uv=False)
cos=np.abs(D@D.T);np.fill_diagonal(cos,0)
new=D[-1];newt=small[-1]
checks={
 'source_hash_matches':sha(repo/'examples/isolated_glu/coverage.py')==report['source_sha256'],
 'hessians_hash_matches':sha(public/'hessians.npz')==report['input_sha256']['hessians.npz'],
 'known_hash_matches':sha(public/'refined-solution.npz')==report['input_sha256']['refined-solution.npz'],
 'monitor_report_matches':all(monitor[k]==v for k,v in report.items()),
 'finite':bool(np.isfinite(D).all()),'count64':D.shape==(64,32),
 'prefix_exactly_preserved':bool(np.array_equal(D[:63],known)),
 'reported_new_loss_agrees':abs(loss[-1]-report['new_candidates'][0]['original_subspace_loss'])<1e-22,
 'reported_new_tangent_agrees':bool(np.allclose(newt,report['new_candidates'][0]['tangent_smallest_three'],rtol=1e-6,atol=1e-14)),
 'no_truth_claims':all(report[k] is False for k in ['sealed','formal_v2_overall_passed','original_weights_recovered','full_glu_recovered','coverage_certified']),
}
out={'scope':'Independent public-array diagnostics only; no fitting and no private/heldout/teacher arrays','checks':checks,
 'source_sha256':report['source_sha256'],'output_sha256':sha(result/'coverage-solution.npz'),
 'count':len(D),'max_unit_norm_error':float(np.max(np.abs(np.linalg.norm(D,axis=1)-1))),
 'max_pairwise_absolute_cosine':float(np.max(cos)),
 'new_max_absolute_cosine_known':float(np.max(np.abs(known@new))),
 'new_original_V_rank1_loss':loss[-1], 'all_original_V_rank1_loss_max':max(loss),
 'new_tangent_smallest_three':newt,'known_second_tangent_max':float(np.max(np.array(small)[:63,1])),
 'tangent_isometry_error_max':max(isometry),'V_isometry_error':float(np.linalg.norm(Q.T@Q-np.eye(128))),
 'component_rank_at_1e-8_relative':int(np.sum(cs>1e-8*cs[0])),
 'component_smallest_singular':float(cs[-1]),
 'original_hessian_rank128_tail_ratio':float(s[128]/s[127]),
 'hessian_top128_condition':float(s[0]/s[127]),
 'teacher_identity_verified':False,'functional_recovery_verified':False,'v2_outcome_changed':False,
 'elapsed_seconds':time.monotonic()-start,
 'own_process_ru_maxrss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)),
 'run_monitor_reported_peak_rss_bytes':monitor['peak_sampled_rss_bytes'],
 'run_monitor_reported_controller_seconds':monitor['controller_seconds']}
print(json.dumps(out,indent=2,allow_nan=False))
