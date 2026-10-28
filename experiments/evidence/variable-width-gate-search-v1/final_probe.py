"""Independent literal controls only; no registered inputs, oracle or data generation."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');sys.path[:0]=[str(R/'examples/isolated_glu'),str(R/'src')]
import variable_gate_search as m
from variable_gate_view import GateJob
class Sink:
 def __init__(self,fail=False):self.data={};self.fail=fail
 def write(self,name,arrays):
  if self.fail and 'refine-chunk' in name:raise OSError('literal flush fault')
  assert name not in self.data
  assert all(not x.flags.writeable for x in arrays.values())
  self.data[name]={k:v.copy() for k,v in arrays.items()}
checks=[];b=np.eye(6)[[0,1,3,4]];job=GateJob('job00',b);x=np.eye(3)[:2]
def init(j,rng,sink):
 m._emit(sink,j.job_id+'-initial-raw',raw=x)
 m._emit(sink,j.job_id+'-initial-starts',starts=x)
 return x.copy(),x.copy()
for flush_fail in (False,True):
 s=Sink(flush_fail);counter=[0]
 def refine(x,b,steps):
  counter[0]+=1
  if counter[0]==2:raise FloatingPointError('literal refiner2 fault')
  return x.copy()
 try:m._search_fixture(job,np.random.default_rng(921),np.random.default_rng(922),s,initializer=init,refiner=refine)
 except FloatingPointError as e:
  assert str(e)=='literal refiner2 fault'
  assert np.array_equal(s.data['job00-initial-pre-refine']['optimized'],x)
  if flush_fail:assert any('flush failed: OSError' in v for v in e.__notes__)
  else:
   z=s.data['job00-initial-refine-chunk000'];assert z['count'][0]==1 and z['attempted'][0]==2 and z['complete'][0]==0
   assert np.array_equal(z['refined'][0],x[0]) and np.all(z['refined'][1:]==0) and z['losses'][0]==0
  checks.append({'control':'refine2_fault_with_flush_failure' if flush_fail else 'refine2_partial_flush','passed':True})
 else:raise AssertionError('injected fault lost')
original=m.np.linalg.eigh;counter=[0];s=Sink()
def eigh(a):
 counter[0]+=1
 if counter[0]==2:raise np.linalg.LinAlgError('literal eig2 fault')
 return original(a)
m.np.linalg.eigh=eigh
try:
 try:m._rescue(np.eye(3)[:1],b,np.random.default_rng(923),s,'small')
 except np.linalg.LinAlgError as e:
  assert str(e)=='literal eig2 fault'
  assert s.data['small-mix-pre000']['raw'].shape==(8,2)
  z=s.data['small-eigen-chunk000'];assert z['count'][0]==1 and z['attempted'][0]==2 and z['complete'][0]==0
  assert np.linalg.norm(z['starts'][0])>0 and np.all(z['starts'][2:]==0)
  checks.append({'control':'eigh2_partial_flush','passed':True})
 else:raise AssertionError('injected eig fault lost')
finally:m.np.linalg.eigh=original
# One real tiny rescue pass, independently inspect all saved fixed chunks/eigenpairs.
s=Sink();rng=np.random.default_rng(924);starts,geo=m._rescue(np.eye(3)[:1],b,rng,s,'small')
assert starts.shape==(512,3) and geo['component_rank']==2 and geo['residual_dimension']==2
raw=np.vstack([s.data[f'small-mix-pre{i:03d}']['raw'] for i in range(32)])
coeff=np.vstack([s.data[f'small-mix-pre{i:03d}']['normalized'] for i in range(32)])
vals=np.vstack([s.data[f'small-eigen-chunk{i:03d}']['values'] for i in range(32)])
vecs=np.concatenate([s.data[f'small-eigen-chunk{i:03d}']['vectors'] for i in range(32)])
indices=np.vstack([s.data[f'small-eigen-chunk{i:03d}']['indices'] for i in range(32)])
actualstarts=np.vstack([s.data[f'small-eigen-chunk{i:03d}']['starts'] for i in range(32)])
assert np.array_equal(starts,actualstarts) and np.array_equal(raw,s.data['small-mixtures']['raw']) and np.array_equal(coeff,s.data['small-mixtures']['normalized'])
assert np.array_equal(vals,s.data['small-eigen']['values']) and np.array_equal(vecs,s.data['small-eigen']['vectors']) and np.array_equal(indices,s.data['small-eigen']['indices'])
reference_rng=np.random.default_rng(924);reference_raw=np.stack([reference_rng.normal(size=2) for _ in range(256)])
assert np.array_equal(raw,reference_raw) and rng.bit_generator.state==reference_rng.bit_generator.state
reference_coeff=np.stack([a/np.linalg.norm(a) for a in raw]);normdiff=float(np.max(abs(coeff-reference_coeff)));assert normdiff<4e-16
res=s.data['small-geometry']['residual'];maxeig=0.;maxorth=0.
for i,(w,v,ix) in enumerate(zip(vals,vecs,indices)):
 packed=coeff[i]@res;matrix=np.zeros((3,3));k=0
 for row in range(3):
  for col in range(row,3):
   matrix[row,col]=matrix[col,row]=packed[k]/(1 if row==col else 2**.5);k+=1
 maxeig=max(maxeig,float(np.max(abs(matrix@v-v*w))))
 maxorth=max(maxorth,float(np.max(abs(v.T@v-np.eye(3)))))
 assert np.array_equal(ix,np.argsort(abs(w))[-2:]) and np.array_equal(starts[2*i:2*i+2],v[:,ix.astype(int)].T)
for i in range(32):
 z=s.data[f'small-eigen-chunk{i:03d}'];assert z['count'][0]==z['attempted'][0]==8 and z['complete'][0]==1
assert maxeig<1e-14 and maxorth<1e-14
checks.append({'control':'512tiny_rescue_starts_32chunks_vs_aggregate_and_independent_eigen_equations','passed':True,'normalization_max_difference_from_frozen_vector_reduction':normdiff,'max_eigen_equation_residual':maxeig,'max_eigenvector_orthogonality_error':maxorth})
# Independent two-step Adam for exact quartic residual in n2 literal geometry.
x0=np.array([[.8,.6]]);ref=x0.copy();mom=np.zeros_like(ref);var=np.zeros_like(ref)
for t in (1,2):
 grad=np.array([[0.,4*ref[0,1]**3]]);grad-=np.sum(grad*ref,axis=1,keepdims=True)*ref
 mom=.9*mom+.1*grad;var=.999*var+.001*grad**2
 ref-=.02*(mom/(1-.9**t))/(np.sqrt(var/(1-.999**t))+1e-8);ref/=np.sqrt(np.sum(ref*ref,axis=1,keepdims=True))
y=m._adam(x0,np.eye(3)[:2],steps=2);adamdiff=float(np.max(abs(y-ref)));assert adamdiff<5e-15
checks.append({'control':'independent_quartic_sphere_gradient_two_Adamsteps','passed':True,'max_difference':adamdiff})
# Exact merge boundaries: thresholdinclusive, signinvariantduplicate and no target cap.
a=np.array([[.9995,np.sqrt(1-.9995**2)]]);low=np.nextafter(.9995,0.);a2=np.array([[low,np.sqrt(1-low**2)]])
known=np.array([[1.,0.]])
assert len(m.merge_candidates(known,a,np.array([1e-8]))[0])==1
assert len(m.merge_candidates(known,a2,np.array([1e-8]))[0])==2
assert len(m.merge_candidates(np.empty((0,2)),known,np.array([np.nextafter(1e-8,np.inf)]))[0])==0
checks.append({'control':'exact_cosine_and_loss_merge_boundaries','passed':True})
out={'status':'passed','source_sha256':hashlib.sha256((R/'examples/isolated_glu/variable_gate_search.py').read_bytes()).hexdigest(),'checks':checks,'registered_arrays_jobs_queries_or_model_forward':False,'production_search_job_run':False,'small_control_work':'one n3rescue with512starts,eigh2fault;two n2Adamsteps;literal fault/mergecontrols','resource_full_protocol_approved':False}
p=Path(__file__).with_name('final-probe-results.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
