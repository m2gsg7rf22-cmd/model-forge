from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
r=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');o=Path('/private/tmp/modelforge-hessian-factor-review')
s=importlib.util.spec_from_file_location('factor',r/'examples/isolated_glu/hessian_factor.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rng=np.random.default_rng(309047);rows=[]
for dim in (1,2,4):
 for repeat in range(3):
  p=6;outputs=3;g=rng.normal(size=dim);x=rng.normal(size=(p,dim));y=rng.normal(size=(p,outputs,dim,dim))
  # Independent sigmoid derivatives and explicit scalar-output basis.
  t=x@g;sig=1/(1+np.exp(-t));beta=sig+t*sig*(1-sig);alpha=sig*(1-sig)*(2+t*(1-2*sig))
  design=[]
  for j in range(dim):
   e=np.eye(dim)[j];basis=np.stack([alpha[k]*x[k,j]*np.outer(g,g)+beta[k]*(np.outer(g,e)+np.outer(e,g)) for k in range(p)])
   design.append(basis.ravel())
  A=np.array(design).T;K=A.T@A;T=y.transpose(1,0,2,3).reshape(outputs,-1)@A
  assert np.allclose(m.gram(g,x),K,rtol=1e-12,atol=1e-12)
  assert np.allclose(m.adjoint(g,y,x),T,rtol=1e-12,atol=1e-12)
  L=np.linalg.cholesky(K);W=np.linalg.solve(L,T.T).T;u,sigma,v=np.linalg.svd(W,full_matrices=False)
  C=np.linalg.solve(L.T,((u[:,0]*sigma[0])[:,None]*v[:1]).T).T
  actual,report=m.rankone_update(g,y,x)
  error=float(np.linalg.norm(actual-C)/max(np.linalg.norm(C),1e-30));assert error<1e-11
  explicit=(actual@A.T).reshape(outputs,p,dim,dim).transpose(1,0,2,3)
  objective=float(np.sum((y-explicit)**2));assert np.isclose(objective,report['objective'],rtol=1e-13)
  # Local quadratic objective lower bound from Eckart-Young, including antisymmetric Y.
  lower=float(np.sum(y*y)-np.sum(W*W)+np.sum(sigma[1:]**2));assert np.isclose(objective,lower,rtol=1e-12,atol=1e-12)
  zero,zero_report=m.rankone_update(g,np.zeros_like(y),x);assert np.array_equal(zero,np.zeros_like(zero)) and zero_report['objective']==0
  rows.append({'dimension':dim,'solution_relative_error':error,'objective':objective})
results={'approved':True,'synthetic_cases':9,'zero_residual_cases':9,'max_solution_relative_error':max(v['solution_relative_error'] for v in rows),'explicit_basis_and_cholesky_verified':True,'nonsymmetric_full_frobenius_verified':True,'saved_experiment_reads':0,'teacher_queries':0,'actual_candidate_fits':0,'cases':rows}
(o/'probe-results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))
files=['examples/isolated_glu/hessian_factor.py','tests/test_glu_hessian_factor.py','docs/glu-hessian-factor-primitive.md'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
review={'approved':True,'scope':'synthetic-only fixed-gate one-unit rank-at-most-one quadratic primitive','sources':{p:sha(r/p) for p in files},'blocking_findings':[],'independent_probe':{'path':str(o/'probe.py'),'sha256':sha(o/'probe.py'),'results_sha256':sha(o/'probe-results.json')},'claims_verified':['Full Frobenius Hessian operator, adjoint and closed Gram','Whitened rank-at-most-one local quadratic minimizer for positive-definite well-conditioned Gram','No ridge or pseudoinverse substitution; singular guard','Zero residual may yield rank zero'],'limitations':['Floating-point local solve; no interval-certified global optimality at the conditioning limit.','No actual saved FP32 Hessian or calibration fitting performed.','No full-model convergence, forward-quality or direction-improvement claim.','Future polish needs a separately reviewed frozen objective, unit order, sweep budget, residual retention rule and fresh public validation.','Existing failed quality results and closed private gate unchanged.']}
(o/'source-review.json').write_text(json.dumps(review,indent=2)+'\n');print('review_sha256',sha(o/'source-review.json'))
