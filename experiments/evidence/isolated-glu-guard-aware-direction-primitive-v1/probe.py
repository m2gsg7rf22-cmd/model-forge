from pathlib import Path
import importlib.util,json,hashlib,tempfile
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-guard-aware-direction-review')
def load(name):
 s=importlib.util.spec_from_file_location(name,R/f'examples/isolated_glu/{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('guard_aware_direction');transport=load('evidence_transport');rng=np.random.default_rng(77741);cases=[]
def geometry(d,K,A):
 cand=[np.zeros(2)];det=K[0,0]*K[1,1]-K[0,1]*K[1,0];un=-.5*np.array([K[1,1]*d[0]-K[0,1]*d[1],-K[1,0]*d[0]+K[0,0]*d[1]])/det
 if np.max(A@un,initial=0.)<=1e-12:cand.append(un)
 for a in A:
  normal=np.linalg.norm(a);r=np.array([-a[1],a[0]])/normal
  for ray in [r,-r]:
   if np.max(A@ray,initial=0.)<=1e-12:
    t=max(0.,-float(d@ray)/(2*float(ray@K@ray)));cand.append(t*ray)
 return min(cand,key=lambda z:float(d@z+z@K@z))
for count in [0,1,2,3,5,8]:
 for trial in range(3):
  Q=rng.normal(size=(2,2));K=Q.T@Q+np.eye(2);d=rng.normal(size=2)*2;A=rng.normal(size=(count,2));expected=geometry(d,K,A);z,info=m.cone_direction(np.array([1000.,0.]),d,K,A);actual=np.array(info['unclipped']);error=float(np.linalg.norm(actual-expected));assert error<1e-8
  dual=np.array(info['original_duals']);assert np.max(A@actual,initial=0.)<1e-8 and np.min(dual,initial=0.)>=-1e-8;assert np.linalg.norm(d+2*K@actual+A.T@dual)<1e-8;assert np.max(abs(dual*(A@actual)),initial=0.)<1e-8;assert info['subsets_considered']==2**count
  cases.append({'constraints':count,'geometry_solution_absolute_error':error})
# Empty-interior and redundant normals must not silently omit the full cone.
A=np.array([[1.,0.],[-1.,0.],[2.,0.],[0.,1.],[0.,-1.],[0.,2.],[1.,0.],[-1.,0.] ]);z,info=m.cone_direction(np.array([100.,0.]),np.array([2.,-3.]),np.eye(2),A);assert np.linalg.norm(z)<1e-12 and info['subsets_considered']==256 and info['dependent_subsets']>0
# A tiny residual can conceal a large geometric error for a narrow cone.
near_cases=[]
for epsilon in [1e-14,1e-11]:
 try:
  near_z,near_info=m.cone_direction(np.array([100.,0.]),np.array([0.,-2.]),np.eye(2),np.array([[1.,0.],[-1.,epsilon]]))
 except ValueError:near_cases.append({'epsilon':epsilon,'status':'rejected'})
 else:
  assert np.linalg.norm(near_z)<1e-7,'near-independent cone false optimum accepted'
  near_cases.append({'epsilon':epsilon,'status':'accurate_zero_solution'})
# Original exact dependency stays supported, including non-pairwise redundant rows.
A3=np.array([[1.,0.,1.],[0.,1.,1.],[1.,1.,2.]])
z3,i3=m.cone_direction(np.array([100.,0.,0.]),-2*np.ones(3),np.eye(3),A3)
assert np.allclose(z3,[1/3,1/3,-1/3],atol=1e-13) and i3['dependent_subsets']==1
# Higher-dimensional diagonal/coordinate cone has a separable exact answer.
K7=np.diag(np.arange(1.,8.));d7=np.array([2.,-3.,4.,-5.,6.,-7.,8.]);e7=np.eye(7);A7=np.array([e7[0],-e7[0],e7[1],e7[2],-e7[3],e7[3],e7[4],e7[5]])
expected7=-.5*d7/np.diag(K7);expected7[0]=expected7[3]=0
for j in [1,2,4,5]:expected7[j]=min(expected7[j],0.)
z7,info7=m.cone_direction(np.ones(7)*100.,d7,K7,A7);assert np.allclose(z7,expected7,rtol=1e-12,atol=1e-12) and info7['subsets_considered']==256
# Direct certificate regression: a large positive dual cannot hide another negative component.
checks=m.certificate(np.zeros(2),np.array([1e12,-1.]),np.array([-1e12,1.]),np.eye(2),np.eye(2));assert checks['dual']>m.TOL
# Nonfinite derived norm/scale must be rejected, not produce zero divided by infinity.
try:
 extreme=m.certificate(np.array([7e153]),np.empty(0),np.array([-1.3e154]),np.ones((1,1)),np.empty((0,1)))
except ValueError:pass
else:assert extreme['stationarity']>m.TOL
# A numerical failure to obtain any passing residual check must fail explicitly.
saved_certificate=m.certificate
try:
 m.certificate=lambda *args:{'primal':1.,'dual':1.,'stationarity':1.,'complementarity':1.}
 try:m.cone_direction(np.ones(2),np.ones(2),np.eye(2),np.empty((0,2)))
 except ValueError as exc:assert 'No numerical KKT' in str(exc)
 else:raise AssertionError('uncertified QP result returned')
finally:m.certificate=saved_certificate
# Row scaling and objective scaling leave the primal cone optimum unchanged.
K=np.array([[3.,.7],[.7,2.]]);d=np.array([-3.,1.]);A=np.array([[1.,0.],[0.,-1.] ]);_,i=m.cone_direction(np.array([.03,.04]),d,K,A);out,j=m.cone_direction(np.array([.03,.04]),10*d,10*K,A*np.array([1e6,1e-6])[:,None]);assert np.allclose(i['unclipped'],j['unclipped'],rtol=1e-12,atol=1e-12);assert np.allclose(np.array(i['original_duals'])*np.array([1e-5,1e7]),j['original_duals'],rtol=1e-10,atol=1e-10);assert np.linalg.norm(out)<=.0025*(1+1e-12) and not j['postclip_kkt_claimed']
# Independent hash/array verification of compact evidence transport, no engine rerun.
with tempfile.TemporaryDirectory(dir=O,prefix='transport-') as tmp:
 arrays={'small':rng.normal(size=(13,7)),'scalar':np.array(1.,dtype=np.float64)};output=Path(tmp)/'evidence';summary=transport.save_evidence(output,arrays,{'synthetic':True});assert summary['evidence_sha256']==hashlib.sha256((output/'numeric-evidence.npz').read_bytes()).hexdigest();assert len((output/'summary.json').read_bytes())<=32768
 with np.load(output/'numeric-evidence.npz',allow_pickle=False) as data:
  for k,a in arrays.items():assert np.array_equal(a,data[k])
 assert (output/'attempt.json').exists()
result={'geometry_cases':cases,'max_geometry_error':max(v['geometry_solution_absolute_error'] for v in cases),'all256_empty_interior_duplicate_opposite_case_passed':True,'no_KKT_candidate_failure_branch_verified':True,'near_independent_geometry_cases':near_cases,'general_exact_dependency_supported':True,'seven_dimensional_separable_coordinate_cone_verified':True,'negative_dual_regression_passed':True,'overflow_scale_regression_passed':True,'normalization_and_original_dual_scaling_verified':True,'preclip_only_KKT_and_radial_feasibility_verified':True,'independent_transport_hash_and_arrays_verified':True,'actual_experiment_arrays_read':0,'teacher_queries':0,'actual_fits':0}
(O/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
