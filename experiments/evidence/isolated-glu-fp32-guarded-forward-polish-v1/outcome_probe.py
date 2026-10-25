from pathlib import Path
import hashlib,json,subprocess
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');B=R.parent/'experiments';D=B/'fp32-guarded-forward-polish-v1-20261025-db8fb33';P=D/'candidate';O=Path('/private/tmp/modelforge-guarded-forward-review');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:dict(np.load(p,allow_pickle=False))
r=json.loads((D/'report.json').read_text());review=json.loads((O/'source-review.json').read_text());b=r['stages']['build'];e=r['stages']['evaluate'];collect=r['stages']['collect'];details=b['polish']
assert r['reviewed_sources']==review['sources']
for p,h in review['sources'].items():assert sha(R/p)==h and hashlib.sha256(subprocess.check_output(['git','show',r['source_commit']+':'+p],cwd=R)).hexdigest()==h
assert sha(Path('/private/tmp/run_guarded_forward.py'))==review['controller_sha256']
for stage,name in [('collect','calibration/manifest.json'),('build','candidate/fit-report.json'),('evaluate','candidate/evaluation-report.json')]:
 receipt=json.loads((D/(stage+'-receipt.json')).read_text());assert receipt==r['stages'][stage]
 for k,v in json.loads((D/name).read_text()).items():assert receipt[k]==v
assert r['input_inventory_before']==r['input_inventory_after'] and len(r['input_inventory_before'])==13
for p,h in r['input_inventory_before'].items():assert sha(B/p)==h
old=B/'fp32-fullmap-dev-v1-20261025-8d2d457/public';original=B/'fp32-cubic-richardson-v1-20261025-188a03a/candidate';paths={'baseline':original/'candidate.npz','fit_report':original/'fit-report.json','old_manifest':old/'manifest.json'}
assert {k:sha(p) for k,p in paths.items()}==b['inputs_before']==b['inputs_after']
assert sha(D/'calibration/manifest.json')==collect['manifest_sha256']==b['forward_manifest_sha256'];assert sha(D/'calibration/observations.npz')==collect['raw_sha256']==b['forward_raw_sha256'];assert collect['vector_calls']==2048 and collect['channel']=='fp32_compute'
assert collect['baseline_sha256']==b['baseline_sha256']==sha(paths['baseline'])==sha(P/'baseline.npz')
assert sha(P/'public/manifest.json')==b['calibration_manifest_sha256']==sha(paths['old_manifest'])
prior=json.loads(paths['fit_report'].read_text());assert prior['candidate_sha256']==b['baseline_sha256'] and prior['calibration_manifest_sha256']==b['calibration_manifest_sha256'] and prior['candidate_built']
assert sha(P/'polish-arrays.npz')==b['arrays_sha256'];assert sha(P/'candidate.npz')==b['candidate_sha256']==e['candidate_sha256'];assert sha(P/'public-validation.npz')==e['validation_sha256']
a=load(paths['baseline']);c=load(P/'candidate.npz');obs=load(D/'calibration/observations.npz');arr=load(P/'polish-arrays.npz');val=load(P/'public-validation.npz');g=a['gates'];assert g.tobytes()==c['gates'].tobytes()
for name,seed in [('train',882030),('guard',882031)]:
 rng=np.random.default_rng(seed);expected=np.vstack([rng.normal(size=(256,32))*scale for scale in [.01,1.,10.,100.] ]);assert np.array_equal(expected,obs[name+'_x']);y=obs[name+'_y'];assert y.shape==(1024,32) and np.isfinite(y).all() and np.array_equal(y,y.astype(np.float32).astype(np.float64))
tx,ty,gx,gy=(obs[k] for k in ['train_x','train_y','guard_x','guard_y']);norms=np.array([np.linalg.norm(ty[k*256:(k+1)*256]) for k in range(4)]);w=np.repeat(1/norms,256);assert np.array_equal(w,arr['weights']) and np.array_equal(ty*w[:,None],arr['weighted_target']);assert np.allclose(norms,details['train_group_norms'],rtol=1e-15)
def unit(i,C,x):
 t=x@g[i];return (t*.5*(1+np.tanh(t*.5)))[:,None]*(x@C.T)
def prediction(C,x):
 t=x@g.T;return np.einsum('nw,wod,nd->no',t*.5*(1+np.tanh(t*.5)),C,x,optimize=True)
def cost(y):return float(np.sum(y*y))
def close(v,exp):assert np.isclose(v,exp,rtol=1e-9,atol=1e-14),(v,exp)
def group(y,ref,rows):return [float(np.linalg.norm(y[k*rows:(k+1)*rows])/np.linalg.norm(ref[k*rows:(k+1)*rows])) for k in range(4)]
def rank1(C):
 assert np.isfinite(C).all();s=np.linalg.svd(C,compute_uv=False);assert s[0]>0 and s[1]<1e-10*s[0]
res=arr['weighted_target'].copy();gr=gy.copy()
for i in range(64):res-=unit(i,a['couplings'][i],tx)*w[:,None];gr-=unit(i,a['couplings'][i],gx)
initialerr=float(np.linalg.norm(res-arr['initial_train_residual'])/np.linalg.norm(res));assert initialerr<1e-10;assert np.allclose(gr,arr['initial_guard_residual'],rtol=1e-9,atol=1e-10)
base=cost(res);close(base,details['initial_cost']);scores=[];maxerror=0.;assert arr['screen'].shape==(64,32,32) and arr['sequential'].shape==(8,32,32)
for i,row in enumerate(details['screening']):
 assert row['unit']==i;rank1(arr['screen'][i]);partial=res+unit(i,a['couplings'][i],tx)*w[:,None];proposed=cost(partial-unit(i,arr['screen'][i],tx)*w[:,None]);close(proposed,row['initial_proposed_cost']);close(base-proposed,row['initial_decrease']);scores.append(base-proposed);maxerror=max(maxerror,abs(proposed-row['initial_proposed_cost']))
selected=[i for i in sorted(range(64),key=lambda i:(-scores[i],i)) if scores[i]>max(base*1e-12,1e-24)][:8];assert selected==details['selected_units'];current=a['couplings'].copy();accepted=[]
for pos,(i,row) in enumerate(zip(selected,details['decisions'])):
 assert row['unit']==i;proposal=arr['sequential'][pos];rank1(proposal);before=cost(res);partial=res+unit(i,current[i],tx)*w[:,None];nextres=partial-unit(i,proposal,tx)*w[:,None];nextgr=gr+unit(i,current[i],gx)-unit(i,proposal,gx);after=cost(nextres);gb=group(gr,gy,256);ga=group(nextgr,gy,256);margin=max(before*1e-12,1e-24);decreased=after<before-margin;guard=all(x<=y+1e-10 for x,y in zip(ga,gb));retain=decreased and guard
 close(before,row['old_cost']);close(after,row['proposed_cost']);close(margin,row['required_decrease']);assert np.allclose(gb,row['guard_before'],rtol=1e-9,atol=1e-13) and np.allclose(ga,row['guard_after'],rtol=1e-9,atol=1e-13);assert (decreased,guard,retain)==(row['training_decreased'],row['guard_passed'],row['accepted'])
 if retain:current[i]=proposal;res=nextres;gr=nextgr;accepted.append(i)
assert accepted==details['accepted_units'] and np.array_equal(current,c['couplings'])
for C in current:rank1(C)
freshtrain=(ty-prediction(current,tx))*w[:,None];freshguard=gy-prediction(current,gx);trainerr=float(np.linalg.norm(freshtrain-arr['final_train_residual'])/np.linalg.norm(freshtrain));guarderr=float(np.linalg.norm(freshguard-arr['final_guard_residual'])/np.linalg.norm(freshguard));assert trainerr<1e-10 and guarderr<1e-10;close(cost(freshtrain),details['final_cost']);assert details['local_rankone_solves']==72 and details['sweeps']==1 and details['guard_is_calibration'] and b['new_calibration_calls']==0 and b['additional_calibration_calls_used']==2048 and b['direction_updates']==0
rng=np.random.default_rng(941030);expected=np.vstack([rng.normal(size=(512,32))*scale for scale in [.01,1.,10.,100.] ]);assert np.array_equal(expected,val['inputs']);assert all(x.shape==(2048,32) and np.isfinite(x).all() for x in val.values());assert np.array_equal(val['reference'],val['reference'].astype(np.float32).astype(np.float64))
errors={}
for name,C in [('prediction',current),('baseline_prediction',a['couplings'])]:
 errors[name]=float(np.linalg.norm(prediction(C,val['inputs'])-val[name])/np.linalg.norm(val[name]));assert errors[name]<1e-12
metrics={}
for name,records in [('prediction',e['groups']),('baseline_prediction',e['baseline_groups'])]:
 metrics[name]=[]
 for k,row in enumerate(records):
  sl=slice(512*k,512*(k+1));delta=val[name][sl]-val['reference'][sl];rel=float(np.linalg.norm(delta)/np.linalg.norm(val['reference'][sl]));absolute=float(abs(delta).max());close(rel,row['relative_frobenius']);close(absolute,row['max_absolute']);metrics[name].append({'scale':row['scale'],'relative_frobenius':rel,'max_absolute':absolute})
assert all(n['relative_frobenius']<o['relative_frobenius'] for n,o in zip(metrics['prediction'],metrics['baseline_prediction']));assert all(row['relative_frobenius']>.001 for row in metrics['prediction']);assert not e['all_groups_relative_at_most_1e_minus3'] and e['paired_same_reference'] and e['vector_calls']==2048
for stage,marker in [('calibration','collection-attempt.json'),('candidate','polish-attempt.json'),('candidate','evaluation-attempt.json')]:assert json.loads((D/stage/marker).read_text())['automatic_retry'] is False
result={'approved':True,'scope':'saved-public-array audit only; paired finite improvement with quality FAILED','report_sha256':sha(D/'report.json'),'source_commit':r['source_commit'],'source_and_dependency_count':12,'old_inventory_count_unchanged':13,'baseline_inputs_unchanged':b['inputs_before'],'forward_raw_sha256':b['forward_raw_sha256'],'forward_manifest_sha256':b['forward_manifest_sha256'],'candidate_sha256':b['candidate_sha256'],'raw_train_guard_seeds_verified':[882030,882031],'raw_calibration_calls_reported':2048,'weights_applied_once_exact':True,'all64_proposal_costs_verified':True,'max_screen_cost_absolute_discrepancy':maxerror,'selected_and_accepted':selected,'all8_training_and_guard_decisions_verified':True,'initial_cost':base,'final_cost':cost(freshtrain),'relative_training_cost_reduction':1-cost(freshtrain)/base,'final_train_residual_relative_discrepancy':trainerr,'final_guard_residual_relative_discrepancy':guarderr,'gates_byte_identical':True,'all_couplings_finite_nonzero_rankone':True,'independent_forward_discrepancy':errors,'paired_same_point_metrics':metrics,'all4_groups_finite_paired_improvement':True,'all4_groups_quality_failed':True,'fixed_quality_threshold':.001,'evaluation_seed_consumed':941030,'evaluation_points':2048,'guard_is_calibration':True,'reviewer_queries':0,'reviewer_new_coupling_solves':0,'old_evaluation_arrays_read':False,'resources':{'controller_seconds':r['controller_seconds'],'max_sampled_worker_tree_rss_bytes':r['max_worker_tree_sampled_rss_bytes'],'samples_by_stage':{k:v['memory_samples'] for k,v in r['stages'].items()},'actual_peak_known':False},'quality':'FAILED','private_gate':'closed','limitations':['Finite paired comparison on one owned public block, not universal or sealed equivalence.','Guard selects updates and is calibration.','Public941030 now consumed; no retry or selection from final evaluation.','Two samples per stage do not prove actual peak RSS or a continuous hard bound.']}
(O/'outcome-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('review_sha256',sha(O/'outcome-review.json'))
