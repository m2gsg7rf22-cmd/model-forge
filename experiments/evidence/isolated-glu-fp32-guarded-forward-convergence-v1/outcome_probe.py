from pathlib import Path
import hashlib,json,subprocess
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');B=R.parent/'experiments';D=B/'fp32-guarded-forward-convergence-v1-20261025-4da51d4';P=D/'candidate';O=Path('/private/tmp/modelforge-forward-convergence-review');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:dict(np.load(p,allow_pickle=False))
r=json.loads((D/'report.json').read_text());review=json.loads((O/'source-review.json').read_text());b=r['stages']['build'];e=r['stages']['evaluate'];details=b['convergence']
assert r['reviewed_sources']==review['sources']
for p,h in review['sources'].items():assert sha(R/p)==h and hashlib.sha256(subprocess.check_output(['git','show',r['source_commit']+':'+p],cwd=R)).hexdigest()==h
assert sha(Path('/private/tmp/run_forward_convergence.py'))==review['controller_sha256']
for stage,name in [('build','fit-report.json'),('evaluate','evaluation-report.json')]:
 receipt=json.loads((D/(stage+'-receipt.json')).read_text());assert receipt==r['stages'][stage]
 for k,v in json.loads((P/name).read_text()).items():assert receipt[k]==v
assert r['input_inventory_before']==r['input_inventory_after'] and len(r['input_inventory_before'])==13
for p,h in r['input_inventory_before'].items():assert sha(B/p)==h
priorroot=B/'fp32-guarded-forward-polish-v1-20261025-db8fb33';old=B/'fp32-fullmap-dev-v1-20261025-8d2d457/public';paths={'baseline':priorroot/'candidate/candidate.npz','fit_report':priorroot/'candidate/fit-report.json','old_manifest':old/'manifest.json','forward_manifest':priorroot/'calibration/manifest.json','observations':priorroot/'calibration/observations.npz'}
assert {k:sha(p) for k,p in paths.items()}==b['inputs_before']==b['inputs_after'];previous=json.loads(paths['fit_report'].read_text());meta=json.loads(paths['forward_manifest'].read_text())
assert previous['candidate_sha256']==b['baseline_sha256']==sha(P/'baseline.npz') and previous['forward_manifest_sha256']==b['forward_manifest_sha256']==sha(paths['forward_manifest']) and previous['forward_raw_sha256']==b['forward_raw_sha256']==sha(paths['observations'])
assert previous['method']=='isolated-glu-fp32-guarded-forward-polish-v1' and previous['candidate_built'] and meta['raw_sha256']==b['forward_raw_sha256'] and meta['baseline_sha256']==previous['baseline_sha256'];assert sha(P/'public/manifest.json')==b['calibration_manifest_sha256']
for p,h in b['evidence_files'].items():assert sha(P/p)==h
assert sha(P/'candidate.npz')==b['candidate_sha256']==e['candidate_sha256'] and sha(P/'public-validation.npz')==e['validation_sha256']
a=load(paths['baseline']);candidate=load(P/'candidate.npz');obs=load(paths['observations']);arr=load(P/'residuals-weights.npz');val=load(P/'public-validation.npz');g=a['gates'];assert g.tobytes()==candidate['gates'].tobytes()
for name,seed in [('train',882030),('guard',882031)]:
 rng=np.random.default_rng(seed);assert np.array_equal(obs[name+'_x'],np.vstack([rng.normal(size=(256,32))*s for s in [.01,1.,10.,100.]]));y=obs[name+'_y'];assert y.shape==(1024,32) and np.isfinite(y).all() and np.array_equal(y,y.astype(np.float32).astype(np.float64))
tx,ty,gx,gy=(obs[k] for k in ['train_x','train_y','guard_x','guard_y']);norms=np.array([np.linalg.norm(ty[k*256:(k+1)*256]) for k in range(4)]);w=np.repeat(1/norms,256);assert np.array_equal(w,arr['weights']) and np.array_equal(ty*w[:,None],arr['weighted_target']) and np.allclose(norms,details['train_group_norms'],rtol=1e-15)
def unit(i,C,x):
 t=x@g[i];return (t*.5*(1+np.tanh(t*.5)))[:,None]*(x@C.T)
def prediction(C,x):
 t=x@g.T;return np.einsum('nw,wod,nd->no',t*.5*(1+np.tanh(t*.5)),C,x,optimize=True)
def cost(y):return float(np.sum(y*y))
def close(v,exp):assert np.isclose(v,exp,rtol=1e-8,atol=1e-14),(v,exp)
def groups(y,ref,rows):return [float(np.linalg.norm(y[k*rows:(k+1)*rows])/np.linalg.norm(ref[k*rows:(k+1)*rows])) for k in range(4)]
def rankcheck(C,nonzero=False):
 assert np.isfinite(C).all();s=np.linalg.svd(C,compute_uv=False);assert s[1]<=1e-10*s[0]
 if nonzero:assert s[0]>0
current=a['couplings'].copy();res=arr['weighted_target'].copy();gr=gy.copy()
for i in range(64):res-=unit(i,current[i],tx)*w[:,None];gr-=unit(i,current[i],gx)
assert np.allclose(res,arr['initial_train_residual'],rtol=1e-9,atol=1e-14) and np.allclose(gr,arr['initial_guard_residual'],rtol=1e-9,atol=1e-10)
initialcost=cost(res);close(initialcost,details['initial_cost']);initialguard=groups(gr,gy,256);assert np.allclose(initialguard,details['initial_guard_relative'],rtol=1e-10,atol=1e-13)
counts=[];maxerror=0.;all_screen=0;all_seq=0;acceptedtotal=0;sweepresults=[]
for k,sweep in enumerate(details['sweeps']):
 assert sweep['sweep']==k;props=load(P/f'sweep-{k}.npz');base=cost(res);close(base,sweep['initial_cost']);assert props['screen'].shape==(64,32,32);scores=[]
 for i,row in enumerate(sweep['screening']):
  assert row['unit']==i;proposal=props['screen'][i];rankcheck(proposal);partial=res+unit(i,current[i],tx)*w[:,None];newcost=cost(partial-unit(i,proposal,tx)*w[:,None]);close(newcost,row['initial_proposed_cost']);close(base-newcost,row['initial_decrease']);maxerror=max(maxerror,abs(newcost-row['initial_proposed_cost']));scores.append(base-newcost);all_screen+=1
 selected=[i for i in sorted(range(64),key=lambda i:(-scores[i],i)) if scores[i]>max(base*1e-12,1e-24)];assert selected==sweep['selected_units'] and len(selected)<=64;assert props['sequential'].shape==(len(selected),32,32);accepted=[]
 for pos,(i,row) in enumerate(zip(selected,sweep['decisions'])):
  assert row['unit']==i;proposal=props['sequential'][pos];rankcheck(proposal);oldcost=cost(res);partial=res+unit(i,current[i],tx)*w[:,None];nextres=partial-unit(i,proposal,tx)*w[:,None];nextgr=gr+unit(i,current[i],gx)-unit(i,proposal,gx);newcost=cost(nextres);gb=groups(gr,gy,256);ga=groups(nextgr,gy,256);margin=max(oldcost*1e-12,1e-24);decreased=newcost<oldcost-margin;guard=all(n<=c+1e-10 and n<=a+1e-10 for n,c,a in zip(ga,gb,initialguard));retain=decreased and guard
  close(oldcost,row['old_cost']);close(newcost,row['proposed_cost']);close(margin,row['required_decrease']);assert np.allclose(gb,row['guard_before'],rtol=1e-9,atol=1e-13) and np.allclose(ga,row['guard_after'],rtol=1e-9,atol=1e-13);assert (decreased,guard,retain)==(row['training_decreased'],row['guard_passed'],row['accepted']);all_seq+=1
  if retain:current[i]=proposal;res=nextres;gr=nextgr;accepted.append(i)
 assert accepted==sweep['accepted_units'];close(cost(res),sweep['final_cost']);assert len(selected)+64==sweep['local_rankone_solves'];counts.append(len(accepted));acceptedtotal+=len(accepted);sweepresults.append({'index':k,'screened':64,'selected':len(selected),'accepted':len(accepted),'final_cost':cost(res)})
assert counts==[32,15,3,2] and acceptedtotal==details['accepted_update_count']==52 and all_screen==256 and all_seq==252 and details['local_rankone_solves']==508 and details['stop_reason']=='max_sweeps' and len(details['sweeps'])==4
assert np.array_equal(current,candidate['couplings'])
for C in current:rankcheck(C,True)
freshtrain=(ty-prediction(current,tx))*w[:,None];freshguard=gy-prediction(current,gx);trainerr=float(np.linalg.norm(freshtrain-arr['final_train_residual'])/np.linalg.norm(freshtrain));guarderr=float(np.linalg.norm(freshguard-arr['final_guard_residual'])/np.linalg.norm(freshguard));assert trainerr<1e-10 and guarderr<1e-10;close(cost(freshtrain),details['final_cost']);assert np.allclose(groups(freshguard,gy,256),details['final_guard_relative'],rtol=1e-9,atol=1e-13)
assert b['new_calibration_calls']==0 and b['reused_calibration_calls']==2048 and b['direction_updates']==0 and details['guard_is_reused_calibration']
rng=np.random.default_rng(941031);assert np.array_equal(val['inputs'],np.vstack([rng.normal(size=(512,32))*s for s in [.01,1.,10.,100.]]));assert all(x.shape==(2048,32) and np.isfinite(x).all() for x in val.values());assert np.array_equal(val['reference'],val['reference'].astype(np.float32).astype(np.float64))
errors={};metrics={}
for name,C,records in [('prediction',current,e['groups']),('baseline_prediction',a['couplings'],e['baseline_groups'])]:
 errors[name]=float(np.linalg.norm(prediction(C,val['inputs'])-val[name])/np.linalg.norm(val[name]));assert errors[name]<1e-12;metrics[name]=[]
 for k,row in enumerate(records):
  sl=slice(k*512,(k+1)*512);delta=val[name][sl]-val['reference'][sl];rel=float(np.linalg.norm(delta)/np.linalg.norm(val['reference'][sl]));absolute=float(abs(delta).max());close(rel,row['relative_frobenius']);close(absolute,row['max_absolute']);metrics[name].append({'scale':row['scale'],'relative_frobenius':rel,'max_absolute':absolute})
assert all(n['relative_frobenius']<o['relative_frobenius'] for n,o in zip(metrics['prediction'],metrics['baseline_prediction']));assert all(n['relative_frobenius']>.001 for n in metrics['prediction']);worsened=[n['scale'] for n,o in zip(metrics['prediction'],metrics['baseline_prediction']) if n['max_absolute']>o['max_absolute']];assert worsened==[.01,1.]
assert e['vector_calls']==2048 and e['paired_same_reference'] and not e['all_groups_relative_at_most_1e_minus3']
marker=json.loads((P/'convergence-attempt.json').read_text());assert marker=={'automatic_retry':False,'max_sweeps':4,'max_local_solves':512,'new_calibration_calls':0};assert json.loads((P/'evaluation-attempt.json').read_text())['automatic_retry'] is False
result={'approved':True,'scope':'independent saved-array public outcome audit; quality FAIL, bounded four-sweep result only','report_sha256':sha(D/'report.json'),'source_commit':r['source_commit'],'sources_verified':12,'unchanged_old_inventory_files':13,'five_inputs_verified':b['inputs_before'],'candidate_sha256':b['candidate_sha256'],'once_weighted_targets_exact':True,'screen_proposals_audited':all_screen,'sequential_proposals_audited':all_seq,'max_screen_cost_absolute_discrepancy':maxerror,'sweeps':sweepresults,'accepted_per_sweep':counts,'accepted_total':52,'reported_local_solves':508,'stop_reason':'max_sweeps','convergence_or_stationary_floor_proven':False,'current_and_initial_guard_bounds_all_verified':True,'initial_cost':initialcost,'final_cost':cost(freshtrain),'relative_training_cost_reduction':1-cost(freshtrain)/initialcost,'independent_final_train_residual_discrepancy':trainerr,'independent_final_guard_residual_discrepancy':guarderr,'gates_byte_identical':True,'couplings_finite_nonzero_rankone':True,'independent_forward_discrepancies':errors,'paired_same_point_metrics':metrics,'all_relative_groups_improved':True,'absolute_maximum_worsened_scales':worsened,'all_four_relative_groups_failed':True,'quality_criterion':.001,'evaluation_seed_consumed':941031,'new_calibration_calls':0,'reported_new_public_evaluation_calls':2048,'reviewer_new_coupling_solves':0,'reviewer_queries':0,'old_evaluation_arrays_read':False,'resources':{'controller_seconds':r['controller_seconds'],'build_sampled_rss_bytes':b['peak_sampled_rss_bytes'],'build_samples':b['memory_samples'],'evaluation_sampled_rss_bytes':e['peak_sampled_rss_bytes'],'evaluation_samples':e['memory_samples'],'actual_peak_known':False},'quality':'FAILED','private_gate':'closed','limitations':['Reused train/guard remain adaptive calibration.','Four-sweep cap and guard vetoes do not prove convergence, stationary floor, or coupling versus direction cause.','Finite paired relative improvement has two absolute-maximum regressions.','New941031 now consumed; no evaluation reuse or unregistered sweep extension.','RSS sampling is not a continuous hard memory-bound proof.']}
(O/'outcome-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('review_sha256',sha(O/'outcome-review.json'))
