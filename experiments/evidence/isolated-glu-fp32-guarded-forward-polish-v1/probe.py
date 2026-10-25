from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-guarded-forward-review');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location('guarded',R/'examples/isolated_glu/guarded_forward.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rng=np.random.default_rng(455911);g=rng.normal(size=(3,3));c=rng.normal(size=(3,2,1))*rng.normal(size=(3,1,3));tx=rng.normal(size=(28,3));gx=rng.normal(size=(28,3));truth=c*1.2
# Independent sigmoid-based full output; no target oracle.
def unit(i,C,x):
 t=x@g[i];return (t/(1+np.exp(-t)))[:,None]*(x@C.T)
def forward(C,x):return sum(unit(i,C[i],x) for i in range(len(g)))
ty=forward(truth,tx);gy=forward(truth,gx);updated,report,arrays=m.polish(g,c,tx,ty,gx,gy,7,2)
norms=np.array([np.linalg.norm(ty[i*7:(i+1)*7]) for i in range(4)]);w=np.repeat(1/norms,7);assert np.array_equal(arrays['weights'],w) and np.array_equal(arrays['weighted_target'],w[:,None]*ty)
res=w[:,None]*(ty-forward(c,tx));gres=gy-forward(c,gx);base=float(np.sum(res**2));scores=[]
for i,row in enumerate(report['screening']):
 cost=float(np.sum((res+w[:,None]*(unit(i,c[i],tx)-unit(i,arrays['screen'][i],tx)))**2));assert np.isclose(cost,row['initial_proposed_cost'],rtol=1e-11,atol=1e-13);scores.append(base-cost)
selected=[i for i in sorted(range(3),key=lambda i:(-scores[i],i)) if scores[i]>max(1e-12*base,1e-24)][:2];assert selected==report['selected_units']
current=c.copy();accepted=[]
def groups(y):return [float(np.linalg.norm(y[k*7:(k+1)*7])/np.linalg.norm(gy[k*7:(k+1)*7])) for k in range(4)]
for pos,(i,row) in enumerate(zip(selected,report['decisions'])):
 oldres=w[:,None]*(ty-forward(current,tx));oldguard=gy-forward(current,gx);proposal=arrays['sequential'][pos];nextcurrent=current.copy();nextcurrent[i]=proposal;newres=w[:,None]*(ty-forward(nextcurrent,tx));newguard=gy-forward(nextcurrent,gx)
 oldcost=float(np.sum(oldres**2));newcost=float(np.sum(newres**2));before=groups(oldguard);after=groups(newguard);guard=all(a<=b+1e-10 for a,b in zip(after,before));decrease=newcost<oldcost-max(1e-12*oldcost,1e-24)
 assert guard==row['guard_passed'] and decrease==row['training_decreased'] and (guard and decrease)==row['accepted']
 assert np.allclose(row['guard_before'],before,rtol=1e-11,atol=1e-14) and np.allclose(row['guard_after'],after,rtol=1e-11,atol=1e-14)
 if guard and decrease:current=nextcurrent;accepted.append(i)
assert np.array_equal(current,updated) and accepted==report['accepted_units'];assert np.allclose(arrays['final_train_residual'],w[:,None]*(ty-forward(current,tx)),atol=1e-12)
# One bad guard group must veto a useful training update.
g=np.array([[1.,.2]]);c=np.array([[[.3,.6]]]);tx=rng.normal(size=(24,2));gx=rng.normal(size=(24,2));ty=forward(c,tx);gy=forward(c,gx);gy[:6]*=.7
bad,badreport,_=m.polish(g,c*.7,tx,ty,gx,gy,6,8);assert not badreport['accepted_units'] and np.array_equal(bad,c*.7)
assert badreport['decisions'][0]['training_decreased'] and not badreport['decisions'][0]['guard_passed']
result={'independent_sigmoid_full_residual_checks':True,'weighted_once_exact':True,'all_screen_costs_ranking_and_sequence_verified':True,'one_worsened_guard_group_vetoes':True,'synthetic_selected':selected,'synthetic_accepted':accepted,'actual_calibration_calls':0,'saved_candidate_fits':0,'teacher_queries':0,'evaluation_arrays_read':0}
(O/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
