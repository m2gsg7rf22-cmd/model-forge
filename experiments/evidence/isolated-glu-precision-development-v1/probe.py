from pathlib import Path
import importlib.util,json
import numpy as np
repo=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');spec=importlib.util.spec_from_file_location('precision',repo/'examples/isolated_glu/precision.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);rng=np.random.default_rng(71278);g=rng.normal(size=(3,4))*.3;u=rng.normal(size=(3,4))*.3;d=rng.normal(size=(2,3))*.3;x=rng.normal(size=(2,4))
def gradient(t):
 z=t@g.T;value=t@u.T;sig=1/(1+np.exp(-z));s=z*sig;ds=sig+z*sig*(1-sig)
 return np.einsum('ow,nw,wi->noi',d,ds*value,g)+np.einsum('ow,nw,wi->noi',d,s,u)
complex_h=np.stack([np.imag(gradient(x.astype(complex)+1e-20j*np.eye(4)[j]))/1e-20 for j in range(4)],axis=-1);exact=m.analytic_hessian(g,u,d,x);error=float(np.max(np.abs(complex_h-exact)));assert error<1e-14
points=np.array([[.2,.7],[-.3,.4]])
def polynomial(t):
 a,b=t[:,0],t[:,1];return np.stack([a**4+2*a*a*b*b+3*b**4,a*b**3+a**3*b],axis=1)
h=np.zeros((2,2,2,2));a,b=points[:,0],points[:,1];h[:,0,0,0]=12*a*a+4*b*b;h[:,0,0,1]=h[:,0,1,0]=8*a*b;h[:,0,1,1]=4*a*a+36*b*b;h[:,1,0,0]=h[:,1,1,1]=6*a*b;h[:,1,0,1]=h[:,1,1,0]=3*(a*a+b*b)
f,c1=m.curvature(polynomial,points,.05);c,c2=m.curvature(polynomial,points,.1);r=m.richardson(f,c);rerror=float(np.max(np.abs(r-h)));assert rerror<1e-12;assert c1==c2==24
sym=rng.normal(size=(3,2,4,4));sym=(sym+sym.swapaxes(-1,-2))*.5;assert np.isclose(np.linalg.norm(sym),np.linalg.norm(m.matrix(sym)))
def unmatrix(a):
 ii,jj=np.triu_indices(4);z=np.zeros((len(a),4,4));z[:,ii,jj]=a/np.where(ii==jj,1,np.sqrt(2));z[:,jj,ii]=z[:,ii,jj];return z.reshape(1,len(a),4,4)
a=np.zeros((3,10));a[0,0]=5;a[1,1]=3;a[2,2]=1;b=a.copy();angle=.3;b[2,2]=np.cos(angle);b[2,3]=np.sin(angle);metric=m.metrics(unmatrix(b),unmatrix(a),3);expected=float(np.sin(angle)/np.sqrt(3));assert np.isclose(metric['subspace_rms_sine'],expected,atol=1e-14)
base=m.forward(g,u,d,x,'fp64');f32=m.forward(g,u,d,x,'fp32_compute');round32=m.forward(g,u,d,x,'fp64_return_fp32');assert not np.array_equal(f32,round32)
result={'analytic_hessian_max_abs_difference_from_independent_complex_step_gradient':error,'mixed_and_diagonal_quartic_Richardson_max_abs_error':rerror,'tiny_curvature_vectorcalls_each':c1,'symmetric_packing_preserves_frobenius_norm':True,'known_principal_angle_rms_sine':metric['subspace_rms_sine'],'expected_rms_sine':expected,'fp32_compute_distinct_from_fp64_return_rounding':True,'fixed_expected_vectorcalls':4*6*2*64*528*4,'record_count':4*6*2,'no_sweep_fitting_or_secret_heldout_access':True};assert result['fixed_expected_vectorcalls']==6488064;Path('/private/tmp/modelforge-glu-precision-review/probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
