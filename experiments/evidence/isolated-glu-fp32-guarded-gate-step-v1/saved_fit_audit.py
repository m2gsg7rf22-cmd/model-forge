from pathlib import Path
import hashlib,json,subprocess
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');B=R.parent/'experiments';D=B/'fp32-guarded-gate-step-v1-20261025-3d3b622';P=D/'candidate';O=Path('/private/tmp/modelforge-guarded-gate-step-review');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:dict(np.load(p,allow_pickle=False))
r=json.loads((D/'report.json').read_text());review=json.loads((O/'source-review.json').read_text());fit=json.loads((P/'fit-report.json').read_text());details=fit['gate_step'];assert r['status']=='failed' and 'output/log budget' in r['error'] and r['stages']=={}
assert not (P/'evaluation-attempt.json').exists() and not (P/'evaluation-report.json').exists() and not (P/'public-validation.npz').exists()
assert r['reviewed_sources']==review['sources']
for p,h in review['sources'].items():assert sha(R/p)==h and hashlib.sha256(subprocess.check_output(['git','show',r['source_commit']+':'+p],cwd=R)).hexdigest()==h
assert sha(Path('/private/tmp/run_guarded_gate_step.py'))==review['controller_sha256'];assert r['input_inventory_before']==r['input_inventory_after'] and len(r['input_inventory_before'])==13
for p,h in r['input_inventory_before'].items():assert sha(B/p)==h
prior=B/'fp32-guarded-forward-convergence-v1-20261025-4da51d4/candidate';cal=B/'fp32-guarded-forward-polish-v1-20261025-db8fb33/calibration';old=B/'fp32-fullmap-dev-v1-20261025-8d2d457/public';paths={'baseline':prior/'candidate.npz','fit_report':prior/'fit-report.json','old_manifest':old/'manifest.json','forward_manifest':cal/'manifest.json','observations':cal/'observations.npz'}
assert {k:sha(p) for k,p in paths.items()}==fit['inputs_before']==fit['inputs_after'];previous=json.loads(paths['fit_report'].read_text());meta=json.loads(paths['forward_manifest'].read_text())
assert previous['candidate_sha256']==fit['baseline_sha256']==sha(P/'baseline.npz') and previous['forward_manifest_sha256']==fit['forward_manifest_sha256'] and previous['forward_raw_sha256']==fit['forward_raw_sha256']==meta['raw_sha256'];assert previous['method']=='isolated-glu-fp32-guarded-forward-convergence-v1' and meta['channel']=='fp32_compute'
assert sha(P/'public/manifest.json')==fit['calibration_manifest_sha256'];assert sha(P/'candidate.npz')==fit['candidate_sha256']=='2f207b106c8dae1f31348c1b46b4ea4bbd08d1f667a5ddb78fa1c86a8002378b'
for p,h in fit['evidence_files'].items():assert sha(P/p)==h
a=load(paths['baseline']);candidate=load(P/'candidate.npz');obs=load(paths['observations']);ev=load(P/'gate-step-evidence.npz');c=a['couplings'];assert candidate['couplings'].tobytes()==c.tobytes();singular=np.linalg.svd(c,compute_uv=False);assert np.isfinite(c).all() and np.all(singular[:,0]>0) and np.all(singular[:,1]<1e-10*singular[:,0])
for name,seed in [('train',882030),('guard',882031)]:
 rng=np.random.default_rng(seed);assert np.array_equal(obs[name+'_x'],np.vstack([rng.normal(size=(256,32))*s for s in [.01,1.,10.,100.]]));assert np.array_equal(obs[name+'_x'],ev[name+'_x']) and np.array_equal(obs[name+'_y'],ev[name+'_y'])
tx,ty,gx,gy=(obs[k] for k in ['train_x','train_y','guard_x','guard_y']);norms=np.array([np.linalg.norm(ty[k*256:(k+1)*256]) for k in range(4)]);w=np.repeat(1/norms,256);assert np.array_equal(w,ev['weights']) and np.array_equal(ty*w[:,None],ev['weighted_target'])
def forward(g,x):
 t=x@g.T;return np.einsum('nw,wod,nd->no',t*.5*(1+np.tanh(t*.5)),c,x,optimize=True)
def cost(x):return float(np.sum(x*x))
def groups(y):return [float(np.linalg.norm(y[k*256:(k+1)*256])/np.linalg.norm(gy[k*256:(k+1)*256])) for k in range(4)]
def maxima(y):return [float(np.max(abs(y[k*256:(k+1)*256]))) for k in range(4)]
def close(v,x):assert np.allclose(v,x,rtol=2e-8,atol=2e-14),(v,x)
def proposal_check(i,rec,G,res):
 gate=G[i];t=tx@gate;sig=np.exp(-np.logaddexp(0,-t));prime=sig+t*sig*(1-sig);branch=tx@c[i].T;d=-2*((w*prime*np.sum(res*branch,axis=1))@tx);K=(tx.T*((w*prime)**2*np.sum(branch*branch,axis=1)))@tx;K=(K+K.T)*.5
 close(rec['gradient'],d);close(rec['gram'],K);eigen=np.linalg.eigvalsh(K);close(rec['eigenvalues'],eigen);assert eigen[0]>1e-12*eigen[-1]>0
 u=np.array(rec['unclipped']);close(K@u,-.5*d);length=np.linalg.norm(u);delta=u.copy() if length==0 else u*min(1,.05*np.linalg.norm(gate)/length);close(rec['direction'],delta);close(rec['gate'],gate);assert np.linalg.norm(delta)<=.05*np.linalg.norm(gate)*(1+1e-12)
 score=-float(d@delta+delta@K@delta);close(score,rec['score']);return score,delta
G=a['gates'].copy();basepred=forward(G,tx);baseguard=forward(G,gx);res=(ty-basepred)*w[:,None];gr=gy-baseguard;ir=groups(gr);ia=maxima(gr);initialcost=cost(res);close(initialcost,details['initial_cost']);close(ir,details['initial_guard_relative']);close(ia,details['initial_guard_max_absolute']);scores=[]
for i,rec in enumerate(details['screening']):
 assert rec['unit']==i;score,_=proposal_check(i,rec,G,res);scores.append(score)
 for field in ['gradient','gram','eigenvalues','unclipped','direction','gate','score']:close(ev['screen_'+field][i],rec[field])
order=sorted(range(64),key=lambda i:(-scores[i],i));assert order==details['screen_order'];selected=[i for i in order if scores[i]>max(initialcost*1e-12,1e-24)][:8];assert selected==details['selected_units'];pos=0;accepted=[];max_full_discrepancy=0.
for seqpos,(i,rec) in enumerate(zip(selected,details['sequential'])):
 assert rec['unit']==i;_,delta=proposal_check(i,rec,G,res)
 for field in ['gradient','gram','eigenvalues','unclipped','direction','gate','score']:close(ev['sequential_'+field][seqpos],rec[field])
 before=cost(res);cr=groups(gr);ca=maxima(gr);old=G[i].copy()
 for power in range(8):
  row=details['trials'][pos];alpha=2.**-power;assert row['unit']==i and row['alpha']==alpha;trial=old+alpha*delta;close(ev['trial_gates'][pos],trial);newg=G.copy();newg[i]=trial;pred=forward(newg,tx);guardpred=forward(newg,gx);nr=(ty-pred)*w[:,None];ngr=gy-guardpred;after=cost(nr);ar=groups(ngr);aa=maxima(ngr);margin=max(before*1e-12,1e-24);decreased=after<before-margin;passed=all(n<=q+1e-10 and n<=z+1e-10 for n,q,z in zip(ar,cr,ir)) and all(n<=q+1e-10*max(1,z) and n<=z+1e-10*max(1,z) for n,q,z in zip(aa,ca,ia));retain=decreased and passed
  close(before,row['old_cost']);close(after,row['proposed_cost']);close(margin,row['required_decrease']);close(cr,row['guard_before_relative']);close(ar,row['guard_after_relative']);close(ca,row['guard_before_max_absolute']);close(aa,row['guard_after_max_absolute']);assert (decreased,passed,retain)==(row['training_decreased'],row['guard_passed'],row['accepted']);pos+=1
  if retain:G=newg;res=nr;gr=ngr;accepted.append(i);break
assert pos==len(details['trials'])==details['trial_count']==57 and accepted==details['accepted_units']==[45,40];assert details['local_linear_solves']==72 and len(details['screening'])==64 and len(details['sequential'])==8
assert np.allclose(G,candidate['gates'],rtol=1e-12,atol=1e-14) and np.isfinite(G).all() and np.all(np.linalg.norm(G,axis=1)>0);close(cost(res),details['final_cost']);close(groups(gr),details['final_guard_relative']);close(maxima(gr),details['final_guard_max_absolute'])
for k,expected in [('initial_train_prediction',basepred),('initial_guard_prediction',baseguard),('final_train_prediction',forward(G,tx)),('final_guard_prediction',forward(G,gx)),('initial_train_residual',(ty-basepred)*w[:,None]),('initial_guard_residual',gy-baseguard),('final_train_residual',res),('final_guard_residual',gr)]:close(ev[k],expected)
assert fit['new_calibration_calls']==0 and fit['reused_calibration_calls']==2048 and not fit['couplings_updated'];marker=json.loads((P/'gate-step-attempt.json').read_text());assert marker['automatic_retry'] is False and marker['max_trials']==64 and marker['max_local_solves']==72
result={'approved':True,'scope':'independent saved fit audit only, suitable for separately reviewed evaluation-only recovery; quality remains unevaluated','report_sha256':sha(D/'report.json'),'fit_report_sha256':sha(P/'fit-report.json'),'evidence_sha256':sha(P/'gate-step-evidence.npz'),'source_commit':r['source_commit'],'candidate_sha256':fit['candidate_sha256'],'sources_verified':13,'five_inputs_verified':fit['inputs_before'],'old_inventory_unchanged':13,'screening_records_verified':64,'sequential_records_verified':8,'Newton_equations_verified_without_solving':True,'trust_cap_and_scores_verified':True,'trial_count':57,'first_accepted_rule_and_all_trial_cost_guards_verified':True,'selected_units':selected,'accepted_units':accepted,'initial_cost':initialcost,'final_cost':cost(res),'relative_training_cost_reduction':1-cost(res)/initialcost,'couplings_byte_identical_and_finite_nonzero_rankone':True,'cached_and_full_prediction_evidence_verified':True,'evaluation_started':False,'evaluation_seed941032_consumed':False,'original_controller_status':'failed output/log budget','build_worker_receipt_available':False,'build_peak_rss_known':False,'recorded_zero_rss_is_missing_not_zero_memory':True,'original_controller_seconds':r['controller_seconds'],'reviewer_direction_solves':0,'reviewer_refits':0,'reviewer_queries':0,'old_evaluation_arrays_read':False,'private_gate':'closed','limitations':['Saved candidate audit does not repair the failed controller receipt or prove original worker resource peak.','No public final quality measurement exists yet; only evaluation-only recovery may continue after separate review.','Reused guard is adaptive calibration, not independent validation.','No fitter rerun or new direction solve is permitted.']}
(O/'saved-fit-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('audit_sha256',sha(O/'saved-fit-audit.json'))
