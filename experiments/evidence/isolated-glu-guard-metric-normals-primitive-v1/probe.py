from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-guard-metric-normals-review');s=importlib.util.spec_from_file_location('normals',R/'examples/isolated_glu/guard_metric_normals.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rng=np.random.default_rng(337194);G=rng.normal(size=(3,4))*.3;C=rng.normal(size=(3,2,4))*.2;X=rng.normal(size=(28,4));X*=np.repeat([.01,1.,5.,20.],7)[:,None]
def F(g,c,x):
 t=x@g.T;return np.einsum('nw,wod,nd->no',t/(1+np.exp(-t)),c,x,optimize=True)
Y=F(G,C,X)+rng.normal(size=(28,2))*np.repeat([.001,.1,1.,4.],7)[:,None];A,report=m.guard_normals(G,C,X,Y,7,1);assert A.shape==(8,4) and report['logical_plane_count']==8;step=1e-24;gradient_errors=[];jac_errors=[]
for k in range(4):
 block=slice(k*7,(k+1)*7);error=F(G,C,X)[block]-Y[block];J=np.empty((7,2,4));squared=[]
 for j in range(4):
  z=G.astype(complex);z[1,j]+=1j*step;values=F(z,C,X)[block];J[:,:,j]=np.imag(values)/step;squared.append(float(np.imag(np.sum((values-Y[block])**2)/np.sum(Y[block]**2))/step))
 records=report['records'][2*k:2*k+2];r,absrec=records;relative=np.array(r['normal']);expected=np.array(squared);err=float(np.linalg.norm(relative-expected)/np.linalg.norm(expected));assert err<1e-10;gradient_errors.append(err)
 row,out=np.unravel_index(np.argmax(abs(error)),error.shape);assert absrec['row']==7*k+row and absrec['local_row']==row and absrec['output']==out and absrec['group']==k and absrec['exact_maximum'];sign=np.sign(error[row,out]);normal=sign*J[row,out];assert np.allclose(absrec['normal'],normal,rtol=1e-11,atol=1e-12);jac_errors.append(float(np.linalg.norm(np.array(absrec['normal'])-normal)/np.linalg.norm(normal)))
 assert np.allclose(relative,2*np.einsum('pod,po->d',J,error)/np.sum(Y[block]**2),rtol=1e-10,atol=1e-12)
 for record in records:assert np.array_equal(A[record['matrix_row']],record['normal'])
# Independent tied-max example: J rows [1,0] and [0,-1].
g=np.array([.2,.1]);sig=1/(1+np.exp(-g));prime=sig+g*sig*(1-sig);c=np.array([[1/prime[0],-1/prime[1]]]);x=np.eye(2);e=np.array([[1.],[-1.]]);y=F(g[None],c[None],x)-e;tie,info=m.metric_group(g,c,x,y,e,3,31);direction=np.array([-1.,1.]);active=[v for v in info['records'] if v['metric']=='maximum_absolute'];assert [v['row'] for v in active]==[31,32] and [v['output'] for v in active]==[0,0];normals=np.array([v['normal'] for v in active]);assert np.allclose(normals,np.eye(2),rtol=1e-12,atol=1e-12);assert normals[0]@direction<0 and normals[1]@direction>0
h=1e-7;changed=F((g+h*direction)[None],c[None],x)-y;derivative=(np.max(abs(changed))-1)/h;assert np.isclose(derivative,max(normals@direction),rtol=1e-6,atol=1e-7)
near=e.copy();near[1,0]=-(1-5e-13);_,nearinfo=m.metric_group(g,c,x,F(g[None],c[None],x)-near,near,3,31);assert [v['exact_maximum'] for v in nearinfo['records'][1:]]==[True,False]
scaled,scaledinfo=m.metric_group(g,-2.7*c,x,-2.7*y,-2.7*e,3,31);assert np.allclose(scaled[0],tie[0],rtol=1e-12,atol=1e-12) and np.allclose(scaled[1:],2.7*tie[1:],rtol=1e-12,atol=1e-12)
# Computed zero normals preserve logical records but do not enter A.
zero,zeroinfo=m.metric_group(np.array([.2]),np.zeros((1,1)),np.array([[1.],[2.]]),np.ones((2,1)),np.array([[-1.],[-.5]]),0,0);assert zero.shape==(0,1) and len(zeroinfo['records'])==2 and all(v['omitted_reason']=='computed_FP64_zero' for v in zeroinfo['records'])
for ref,error in [(np.array([[1.],[0.]]),np.array([[0.],[1e-200]])),(np.array([[1e150],[0.]]),np.array([[0.],[1e-160]]))]:
 try:m.metric_group(np.array([.2]),np.zeros((1,1)),np.array([[1.],[2.]]),ref,error,0,0)
 except ValueError:pass
 else:raise AssertionError('underflowed nonzero error metric accepted')
try:m.guard_normals(np.array([[.2]]),np.zeros((1,1,1)),np.ones((8,1)),np.ones((8,1)),2,0)
except ValueError as exc:assert 'budget' in str(exc)
else:raise AssertionError('zero omission bypassed logical tie cap')
for args in [(g,c,x,y,e,True,0),(g,c,x,y,e,0,4095)]:
 try:m.metric_group(*args)
 except ValueError:pass
 else:raise AssertionError('bad provenance accepted')
result={'complex_step_full_other_units_selected_unit_gradients_verified':True,'max_relative_gradient_error':max(gradient_errors),'max_signed_Jacobian_error':max(jac_errors),'nonuniform_group_scales_and_rowmajor_provenance_verified':True,'single_tie_plane_counterexample_verified':True,'exact_and_near_flags_verified':True,'negative_response_scaling_verified':True,'computed_zero_records_omitted_without_budget_bypass':True,'nonzero_error_metric_underflows_rejected':True,'provenance_bounds_checked':True,'actual_saved_arrays_read':0,'teacher_queries':0,'actual_fits':0}
(O/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
