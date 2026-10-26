"""Literal hand-sized independent math probes; no RNG, teachers, queries or fit."""
import importlib.util,json,hashlib
from pathlib import Path
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-domain-math-review');s=importlib.util.spec_from_file_location('ind_domain_math',R/'examples/isolated_glu/scaling_domain_math.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
g=np.array([[.7,-.2,.3],[-.1,.5,.4]]);u=np.array([[.2,.8,-.6],[.9,-.3,.1]]);d=np.array([[1.2,-.4],[-.7,.6]]);x=np.array([[.3,-.5,.2],[-.2,.4,.7]])
# Hessian via complex-step of separate explicit Jacobian using logistic exp.
def jac(v):
 z=g@v;uv=u@v;sig=1/(1+np.exp(-z));act=z*sig;prime=sig+z*sig*(1-sig)
 return np.einsum('ow,wi->oi',d,(prime*uv)[:,None]*g+act[:,None]*u)
expected=np.empty((2,2,3,3))
for k in range(2):
 for j in range(3):
  z=x[k].astype(complex);z[j]+=1e-25j;expected[k,:,:,j]=np.imag(jac(z))/1e-25
actual=m.analytic_hessian(g,u,d,x);hessian_error=float(np.max(abs(actual-expected)));assert hessian_error<1e-14
A=np.array([[2.,-3.],[-3.,5.]]);B=np.array([[4.,.2],[.2,-1.]]);assert abs(m.svec(A)@m.svec(B)-np.sum(A*B))<1e-13
# Independently hand-constructed atoms for one unit, ordering gg then gu+ug.
atom=m.symmetric_atoms(np.array([[2.,-1.]]),np.array([[3.,4.]]));target=np.array([[4.,-2*np.sqrt(2),1.],[12.,5*np.sqrt(2),-8.]]);np.testing.assert_allclose(atom,target)
for a,rank,reason in [(np.zeros((2,3)),0,'zero_rank'),(np.eye(3),3,'full_space'),(np.array([[1.,0.,0.]]),1,'zero_next_singular')]:
 sp=m.spectrum(a);assert sp['rank']==rank and sp['gap'] is None and sp['gap_reason']==reason and len(sp['singular_values'])==3
assert m.spectrum(np.diag([1.,1e-10,0.]))['rank']==1;assert m.spectrum(np.diag([1.,np.nextafter(1e-10,np.inf),0.]))['rank']==2
# Empty span distances1/full span0; small representable residual is retained.
v=np.array([[1.,0.],[0.,1.],[3.,4.]]);np.testing.assert_allclose(m.direction_residuals(np.zeros((0,3)),v),1.);np.testing.assert_allclose(m.direction_residuals(np.eye(3),v),0.,atol=1e-15)
tiny=m.direction_residuals(np.array([[1.,1e-200,0.]]),np.array([[1.,0.]]));assert tiny[0]>0 and abs(tiny[0]/1e-200-1)<1e-14
try:m.direction_residuals(np.array([[1.,0.,0.]]),np.array([[1.,1e-200]]))
except ValueError:pass
else:raise AssertionError('outer product underflow silently accepted')
# Four-corner diagonal2h quartic bias8h², cross term3, Richardson exact.
xx=np.array([.5,-.25]);hs=[]
for step in [.125,.25]:
 q=m.four_corner_inputs(xx,step);assert q.shape==(4,3,2)
 responses=np.stack([np.column_stack([t[:,0]**4+3*t[:,0]*t[:,1]+t[:,1]**4]) for t in q]);h=m.curvature_from_responses(responses,2,step)
 np.testing.assert_allclose(h[0],[[12*xx[0]**2+8*step**2,3.],[3.,12*xx[1]**2+8*step**2]],atol=1e-14);hs.append(h)
np.testing.assert_allclose(m.richardson(*hs)[0],[[12*xx[0]**2,3.],[3.,12*xx[1]**2]],atol=1e-14)
# Large/tiny global singular scaling remains finite, does not change rank.
for scale in [1e-200,1e200]:assert m.spectrum(np.eye(3)*scale)['rank']==3
result={'passed':True,'hessian_complexstep_max_absolute_error':hessian_error,'svec_isometry_and_interleaved_atoms':True,'strict_rank_boundary_and_all_gap_reasons':True,'full_zero_deficient_span_controls':True,'tiny_representable_residual':float(tiny[0]),'direction_outer_underflow_failclosed':True,'four_corner_diagonal2h_quartic_and_richardson':True,'global_svd_scaling_extremes':True,'registered_seeds_used':0,'rng_calls':0,'teacher_queries':0,'fitting':0,'actual_or_consumed_data_reads':0}
(P/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
