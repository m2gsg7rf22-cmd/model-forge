from pathlib import Path
import importlib.util,json,hashlib,ast
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-hessian-polish-review')
s=importlib.util.spec_from_file_location('polish',R/'examples/isolated_glu/hessian_polish.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rng=np.random.default_rng(556124);cases=[]
for trial in range(3):
 g=rng.normal(size=(5,3));c=rng.normal(size=(5,2,1))*rng.normal(size=(5,1,3));x=rng.normal(size=(8,3))
 truth=c+rng.normal(size=c.shape)*.1
 h=sum(m.factor.operator(g[i],truth[i],x) for i in range(5))
 def prediction(C):return sum(m.factor.operator(g[i],C[i],x) for i in range(5))
 def solve(i,Y):
  A=np.column_stack([m.factor.operator(g[i],np.eye(3)[j:j+1],x).ravel() for j in range(3)])
  T=Y.transpose(1,0,2,3).reshape(2,-1)@A;L=np.linalg.cholesky(A.T@A);W=np.linalg.solve(L,T.T).T
  u,s,v=np.linalg.svd(W,full_matrices=False);Z=(u[:,0]*s[0])[:,None]*v[:1]
  return np.linalg.solve(L.T,Z.T).T
 initial=h-prediction(c);base=np.sum(initial**2);screen=[]
 for i in range(5):
  partial=initial+m.factor.operator(g[i],c[i],x);p=solve(i,partial);screen.append(float(base-np.sum((partial-m.factor.operator(g[i],p,x))**2)))
 expected=sorted(range(5),key=lambda i:(-screen[i],i));expected=[i for i in expected if screen[i]>max(1e-12*base,1e-24)][:3]
 actual,report,before,after,evidence=m.polish(g,c,h,x,3)
 assert evidence["screen"].shape==c.shape and evidence["sequential"].shape==(len(expected),*c.shape[1:])
 for i,row in enumerate(report["screening"]):
  partial=before+m.factor.operator(g[i],c[i],x)
  observed=float(np.sum((partial-m.factor.operator(g[i],evidence["screen"][i],x))**2))
  assert np.isclose(observed,row["initial_proposed_cost"],rtol=1e-12)
 assert report['selected_units']==expected and report['local_rankone_solves']==5+len(expected)
 assert np.allclose(before,initial,rtol=1e-12,atol=1e-12)
 independent=c.copy();accepted=[]
 for i,decision in zip(expected,report['decisions']):
  residual=h-prediction(independent);old=float(np.sum(residual**2));partial=residual+m.factor.operator(g[i],independent[i],x);p=solve(i,partial);new=float(np.sum((partial-m.factor.operator(g[i],p,x))**2));retain=new<old-max(1e-12*old,1e-24)
  assert decision['accepted']==retain and np.isclose(decision['old_cost'],old,rtol=1e-12)
  if retain:independent[i]=p;accepted.append(i)
 assert accepted==report['accepted_units'];assert np.allclose(independent,actual,rtol=1e-11,atol=1e-12)
 fresh=h-prediction(actual);rel=float(np.linalg.norm(fresh-after)/max(np.linalg.norm(fresh),1e-30));assert rel<1e-12
 cases.append({'selected':expected,'accepted':accepted,'fresh_residual_discrepancy':rel,'initial_cost':report['initial_cost'],'final_cost':report['final_cost']})
# Evaluator differs from frozen prior evaluator only by method and fresh seed.
old=(R/'examples/isolated_glu/cubic_evaluate.py').read_text().replace('isolated-glu-fp32-cubic-richardson-v1','isolated-glu-fp32-hessian-coupling-polish-v1').replace('941028','941029')
assert ast.dump(ast.parse(old))==ast.dump(ast.parse((R/'examples/isolated_glu/hessian_polish_evaluate.py').read_text()))
zero_g=np.array([[1.,.2]]);zero_c=np.array([[[.3,.6]]]);zero_x=np.array([[1.,2.],[-1.,.1]])
_,zero_report,_,_,zero_evidence=m.polish(zero_g,zero_c,m.factor.operator(zero_g[0],zero_c[0],zero_x),zero_x,8)
assert zero_evidence['sequential'].shape==(0,1,2) and zero_evidence['screen'].shape==zero_c.shape
result={'proposal_evidence_shape_and_cost_verified':True,'empty_sequence_shape_verified':True,'cases':cases,'tiny_synthetic_only':True,'actual_saved_hessian_reads':0,'teacher_queries':0,'evaluation_reads':0,'evaluator_ast_matches_only_method_and_seed_change':True}
(O/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
