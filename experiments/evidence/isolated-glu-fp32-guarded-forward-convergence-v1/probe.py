from pathlib import Path
import importlib.util,json,hashlib,ast
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-forward-convergence-review');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location('convergence',R/'examples/isolated_glu/forward_convergence.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rng=np.random.default_rng(774193);g=rng.normal(size=(3,2));c=rng.normal(size=(3,2,1))*rng.normal(size=(3,1,2));truth=c*1.15;block=rng.normal(size=(12,2));tx=np.tile(block,(4,1));gx=tx.copy()
def unit(i,C,x):
 t=x@g[i];return (t/(1+np.exp(-t)))[:,None]*(x@C.T)
def forward(C,x):return sum(unit(i,C[i],x) for i in range(len(g)))
ty=forward(truth,tx);gy=ty.copy();updated,report,arr=m.converge(g,c,tx,ty,gx,gy,12,4);w=arr['weights'];assert np.array_equal(arr['weighted_target'],w[:,None]*ty)
current=c.copy();solves=0;acceptedtotal=0;initialguard=report['initial_guard_relative'];cost=lambda y:float(np.sum(y*y));relative=lambda y:[float(np.linalg.norm(y[i*12:(i+1)*12])/np.linalg.norm(gy[i*12:(i+1)*12])) for i in range(4)]
for k,sweep in enumerate(report['sweeps']):
 res=w[:,None]*(ty-forward(current,tx));gr=gy-forward(current,gx);base=cost(res);scores=[];screen=arr['screen_'+str(k)];seq=arr['sequential_'+str(k)]
 assert screen.shape==c.shape
 for i,row in enumerate(sweep['screening']):
  proposalcost=cost(res+w[:,None]*(unit(i,current[i],tx)-unit(i,screen[i],tx)));assert np.isclose(proposalcost,row['initial_proposed_cost'],rtol=1e-10,atol=1e-14);scores.append(base-proposalcost)
 selected=[i for i in sorted(range(len(g)),key=lambda i:(-scores[i],i)) if scores[i]>max(base*1e-12,1e-24)];assert selected==sweep['selected_units'];assert seq.shape==(len(selected),*c.shape[1:]);accepted=[]
 for pos,(i,row) in enumerate(zip(selected,sweep['decisions'])):
  before=cost(res);nextc=current.copy();nextc[i]=seq[pos];nr=w[:,None]*(ty-forward(nextc,tx));ng=gy-forward(nextc,gx);after=cost(nr);gb=relative(gr);ga=relative(ng);passed=all(n<=b+1e-10 and n<=a+1e-10 for n,b,a in zip(ga,gb,initialguard));decreased=after<before-max(before*1e-12,1e-24)
  assert (decreased,passed,decreased and passed)==(row['training_decreased'],row['guard_passed'],row['accepted']);assert np.allclose(ga,row['guard_after'],rtol=1e-10,atol=1e-13)
  if passed and decreased:current=nextc;res=nr;gr=ng;accepted.append(i)
 assert accepted==sweep['accepted_units'];assert np.isclose(cost(res),sweep['final_cost'],rtol=1e-10,atol=1e-14);solves+=len(g)+len(selected);acceptedtotal+=len(accepted)
assert np.array_equal(current,updated);assert solves==report['local_rankone_solves'] and acceptedtotal==report['accepted_update_count'];assert len(report['sweeps'])==4 and report['stop_reason']=='max_sweeps';assert solves<=4*2*len(g)
assert np.allclose(arr['final_train_residual'],w[:,None]*(ty-forward(updated,tx)),rtol=1e-10,atol=1e-13)
# Guard cannot drift beyond its initial ceiling, even if each current step appears within tolerance.
assert not m.guard_ok([.1+1.5e-10]*4,[.1+1e-10]*4,[.1]*4)
assert not m.guard_ok([.09+2e-10]*4,[.09]*4,[.1]*4)
assert m.guard_ok([.09+5e-11]*4,[.09]*4,[.1]*4)
# A blocked one-unit update stops after the first sweep.
g=np.array([[1.,.2]]);c=np.array([[[.3,.6]]]);tx=rng.normal(size=(20,2));gx=rng.normal(size=(20,2));ty=forward(c,tx);gy=forward(c,gx)*.7
unchanged,blocked,_=m.converge(g,c*.7,tx,ty,gx,gy,5,4);assert blocked['stop_reason']=='zero_accepted' and len(blocked['sweeps'])==1 and blocked['accepted_update_count']==0 and np.array_equal(unchanged,c*.7)
old=(R/'examples/isolated_glu/guarded_forward_evaluate.py').read_text().replace('isolated-glu-fp32-guarded-forward-polish-v1','isolated-glu-fp32-guarded-forward-convergence-v1').replace('941030','941031');assert ast.dump(ast.parse(old))==ast.dump(ast.parse((R/'examples/isolated_glu/forward_convergence_evaluate.py').read_text()))
result={'synthetic_full_four_sweeps_verified':True,'synthetic_solves':solves,'synthetic_accepted':acceptedtotal,'all_proposal_costs_ranking_sequence_and_dual_guards_verified':True,'initial_and_current_guard_ceilings_challenged':True,'zero_accept_first_sweep_stop_verified':True,'evaluator_ast_only_method_seed_changed':True,'actual_saved_data_reads':0,'teacher_queries':0,'saved_candidate_fits':0,'evaluation_reads':0}
(O/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
