from pathlib import Path
import importlib.util,json,hashlib,ast
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-guarded-gate-step-review');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location('gatestep',R/'examples/isolated_glu/guarded_gate_step.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rng=np.random.default_rng(817734);g=np.array([[.4,-.2]]);c=np.array([[[.3,.5],[.6,1.]]]);x=rng.normal(size=(48,2))
def forward(G,X):
 t=X@G.T;return np.einsum('nw,wod,nd->no',t/(1+np.exp(-t)),c,X,optimize=True)
y=forward(g*1.03,x);gy=forward(g*1.01,x);weights,weighted,_=m.core.prepare_weights(y,12);res=weighted-forward(g,x)*weights[:,None]
# Complex-step Jacobian independently supplies weighted J, full gradient and Gram.
eps=1e-24;J=np.column_stack([np.imag(forward(g.astype(complex)+1j*eps*np.eye(2)[j:j+1],x)*weights[:,None]).ravel()/eps for j in range(2)]);d=-2*J.T@res.ravel();K=J.T@J;unclipped=-np.linalg.solve(K,J.T@(-res.ravel()));expected=-.5*np.linalg.solve(K,d);assert np.allclose(unclipped,expected)
delta,info=m.proposal(g[0],c[0],x,weights,res);assert np.allclose(info['gradient'],d,rtol=1e-12,atol=1e-13) and np.allclose(info['gram'],K,rtol=1e-12,atol=1e-13)
length=np.linalg.norm(expected);clipped=expected*min(1,.05*np.linalg.norm(g[0])/length);assert np.allclose(delta,clipped,rtol=1e-12,atol=1e-13);assert np.linalg.norm(delta)<=.05*np.linalg.norm(g[0])*(1+1e-13)
assert np.isclose(info['score'],-(d@delta+delta@K@delta),rtol=1e-12,atol=1e-15)
updated,report,arrays=m.step(g,c,x,y,x,gy,12);assert report['accepted_units']==[0] and report['local_linear_solves']==2
# First full step violates guard; first passing reduced alpha must be accepted, with no later trial.
assert len(report['trials'])>1 and report['trials'][-1]['accepted'] and all(not v['accepted'] for v in report['trials'][:-1]);assert [t['alpha'] for t in report['trials']]==[2.**-i for i in range(len(report['trials']))]
ir=m.core.group_relative(gy-forward(g,x),gy,12);ia=m.maxabs(gy-forward(g,x),12);initialcost=float(np.sum(((y-forward(g,x))*weights[:,None])**2))
for row,trialgate in zip(report['trials'],arrays['trial_gates']):
 trialg=g.copy();trialg[0]=trialgate;trainres=(y-forward(trialg,x))*weights[:,None];guardres=gy-forward(trialg,x);cost=float(np.sum(trainres**2));ar=m.core.group_relative(guardres,gy,12);aa=m.maxabs(guardres,12);passed=all(n<=a+1e-10 for n,a in zip(ar,ir)) and all(n<=a+1e-10*max(1,a) for n,a in zip(aa,ia));decreased=cost<initialcost-max(initialcost*1e-12,1e-24)
 assert passed==row['guard_passed'] and decreased==row['training_decreased'];assert np.isclose(cost,row['proposed_cost'],rtol=1e-10,atol=1e-14)
assert np.allclose(arrays['final_train_prediction'],forward(updated,x),rtol=1e-13,atol=1e-13) and np.array_equal(c,np.array([[[.3,.5],[.6,1.]]]))
# Four independent constraint families: relative/absolute x current/initial.
base=[.1]*4;ab=[10.]*4
assert not m.guard_ok([.10000001]*4,ab,base,ab,[.2]*4,[20.]*4)
assert not m.guard_ok([.10000001]*4,ab,[.2]*4,[20.]*4,base,ab)
assert not m.guard_ok(base,[10.00001]*4,base,ab,[.2]*4,[20.]*4)
assert not m.guard_ok(base,[10.00001]*4,[.2]*4,[20.]*4,base,ab)
old=(R/'examples/isolated_glu/forward_convergence_evaluate.py').read_text().replace('isolated-glu-fp32-guarded-forward-convergence-v1','isolated-glu-fp32-guarded-gate-step-v1').replace('941031','941032');assert ast.dump(ast.parse(old))==ast.dump(ast.parse((R/'examples/isolated_glu/guarded_gate_step_evaluate.py').read_text()))
result={'independent_complex_step_gradient_and_Gram_verified':True,'minus_half_Ksolve_and_trust_cap_verified':True,'first_accepted_trial_alpha':report['trials'][-1]['alpha'],'preceding_rejected_trials':len(report['trials'])-1,'all_trial_costs_and_guards_independently_verified':True,'cached_full_prediction_verified':True,'four_guard_families_challenged':True,'evaluator_ast_only_method_seed_change':True,'actual_saved_candidate_reads':0,'actual_target_queries':0,'actual_fits':0,'old_evaluation_reads':0}
(O/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
