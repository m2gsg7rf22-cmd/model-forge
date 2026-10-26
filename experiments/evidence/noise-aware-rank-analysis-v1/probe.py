"""Literal-only observation-rank core probes: no paths/data/teacher in estimator."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-rank-analysis-review');sp=importlib.util.spec_from_file_location('ind_rank_current',R/'examples/isolated_glu/rank_selection.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
# Declared gap>=10 is assessed before log roundoff at the boundary.
boundary=[]
for top,expected in [(np.nextafter(10.,0.),None),(10.,1),(np.nextafter(10.,np.inf),1)]:
 a=np.diag([top,1.]);r=m.select(a,a,a);assert r['choices']['gap']['rank']==expected and r['choices']['stable']['rank']==expected;boundary.append(expected)
# Exact computed-score ties choose smallest eligible index; masks are strict bool.
assert m.choose(np.array([3.,3.,1.]),np.array([True,True,True]))==0
for mask in [np.array([np.nan,0.]),np.array([1.,0.])]:
 try:m.choose(np.array([1.,2.]),mask)
 except ValueError:pass
 else:raise AssertionError('nonbool mask accepted')
# Independent explicit-projector comparison to cumulative overlap formula.
theta=.005;Q=np.array([[np.cos(theta),-np.sin(theta),0.],[np.sin(theta),np.cos(theta),0.],[0.,0.,1.]])
distance=m._prefix_distance(np.eye(3),Q,2);expected=[]
for k in [1,2]:
 A=np.eye(3)[:k].T@np.eye(3)[:k];B=Q[:k].T@Q[:k];expected.append(np.linalg.norm(A-B)/np.sqrt(2*k))
np.testing.assert_allclose(distance,expected,atol=1e-12,rtol=1e-10)
# Three views agree at a small rotation; unstable top-direction is rejected.
a=np.diag([1.,1e-4,1e-4]);out=[]
for angle,rank in [(.003,1),(.02,None)]:
 q=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
 r=m.select(a,a@q,a@q.T);assert r['choices']['gap']['rank']==1 and r['choices']['stable']['rank']==rank;out.append(rank)
# Fixed floor is now relative to leading singular value, independent of entry basis.
a=np.diag([1.,1.9e-7,0.]);h=1/np.sqrt(2);q=np.array([[h,-h,0.],[h,h,0.],[0.,0.,1.]])
base=m.select(a,a,a);rot=m.select(q@a,q@a,q@a);assert base['choices']==rot['choices']
for scales in [(1e-200,1e200,1.),(1e200,1e-200,1e-100)]:
 r=m.select(*(a*s for s in scales));assert r['choices']==base['choices']
# Identical full-rank spiky signal/common-bias views cannot certify either origin.
observed=np.diag([1.,1e-4,1e-4]);r=m.select(observed,observed,observed);assert r['choices']['strict']['rank']==3 and r['choices']['stable']['rank']==1 and all(not v['certificate'] for v in r['choices'].values())
for a,strict in [(np.zeros((3,3)),0),(np.eye(3),3),(np.array([[1.,0.,0.]]),1),(np.ones((1,1)),1)]:
 r=m.select(a,a,a);assert r['choices']['strict']['rank']==strict and len(r['spectra']['richardson'])==a.shape[1];assert r['certificate'] is False
for bad in [np.array([[np.nan]]),np.ones((2,2),dtype=np.float32),np.ones((1,1),dtype=complex),np.zeros((0,2))]:
 try:m.select(bad,bad,bad)
 except ValueError:pass
 else:raise AssertionError('invalid matrix accepted')
result={'passed':True,'gap10_nextafter_boundary_ranks':boundary,'computed_score_tie_smallest_index':True,'nonbool_nan_masks_rejected':True,'prefix_RMS_against_explicit_projectors_max_error':float(np.max(abs(distance-expected))),'small_and_large_rotation_stable_choices':out,'leading_relative_resolution_orthogonal_entry_basis_invariant_control':True,'independent_view_scale_invariance_1eplusminus200':True,'spiky_full_signal_common_bias_noncertificate':True,'zero_full_rowlimited_p1_controls':True,'source_sha256':hashlib.sha256((R/'examples/isolated_glu/rank_selection.py').read_bytes()).hexdigest(),'actual_saved_observation_reads':0,'registered_seeds_or_RNG':0,'teacher_or_forward_calls':0,'fitting':0}
(P/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
