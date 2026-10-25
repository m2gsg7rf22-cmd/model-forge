from pathlib import Path
import json,hashlib,subprocess
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');B=R.parent/'experiments';O=Path('/private/tmp/modelforge-bottleneck-review');D=B/'fp32-calibration-bottleneck-v1-20261025-9410729';old=B/'fp32-fullmap-dev-v1-20261025-8d2d457/public';new=B/'fp32-cubic-richardson-v1-20261025-188a03a'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((D/'report.json').read_text());inner=json.loads((D/'diagnostic/report.json').read_text());review=json.loads((O/'source-review.json').read_text())
for p,h in review['sources'].items():
 assert sha(R/p)==h
 assert hashlib.sha256(subprocess.check_output(['git','show',report['source_commit']+':'+p],cwd=R)).hexdigest()==h
assert report['reviewed_sources']==review['sources']
assert sha(Path('/private/tmp/run_calibration_bottleneck.py'))==review['controller_sha256']
paths={'multiscale':old/'multiscale.npz','old_manifest':old/'manifest.json','signed':new/'additional-calibration/signed.npz','additional_manifest':new/'additional-calibration/manifest.json','derived':new/'candidate/derived-cubic.npz','candidate':new/'candidate/candidate.npz','fit_report':new/'candidate/fit-report.json'}
actual={k:sha(p) for k,p in paths.items()};assert actual==inner['inputs_before']==inner['inputs_after']
for k,v in inner.items():assert report['result'][k]==v
assert sha(D/'diagnostic/calibration-comparisons.npz')==inner['arrays_sha256']
load=lambda p:dict(np.load(p,allow_pickle=False))
a=load(paths['multiscale']);s=load(paths['signed']);d=load(paths['derived']);candidate=load(paths['candidate']);saved=load(D/'diagnostic/calibration-comparisons.npz')
meta=json.loads(paths['old_manifest'].read_text());add=json.loads(paths['additional_manifest'].read_text());fit=json.loads(paths['fit_report'].read_text())
assert meta['files']['multiscale.npz']==actual['multiscale'] and add['calibration_manifest_sha256']==actual['old_manifest'] and add['raw_sha256']==actual['signed']
for key,target in [('candidate_sha256','candidate'),('derived_rhs_sha256','derived'),('additional_manifest_sha256','additional_manifest'),('calibration_manifest_sha256','old_manifest')]:assert fit[key]==actual[target]
assert np.array_equal(s['small_x'],a['small_x'])
f=(s['fine_pos']-s['fine_neg'])*.5/(.25*.03**3);co=(s['coarse_pos']-s['coarse_neg'])*.5/(.25*.06**3);rhs=(4*f-co)/3
assert np.array_equal(rhs,d['rhs']) and np.array_equal(rhs*(.25*.001**3),d['bridged_small_odd'])
assert np.array_equal(saved['new_rhs'],rhs) and np.array_equal(saved['old_rhs'],a['small_odd']/(.25*.001**3))
g,c=candidate['gates'],candidate['couplings']
def forward_component(x,kind):
 t=x@g.T
 factor={'cubic':lambda:t*t,'even':lambda:.5*t,'odd':lambda:.5*t*np.tanh(.5*t),'sat':lambda:.5*np.abs(t)}[kind]()
 return np.einsum('nw,wod,nd->no',factor,c,x,optimize=True)
e=np.eye(32);canonical=np.vstack([e]+[(e[i]+e[j])[None,:] for i in range(32) for j in range(i+1,32)])
expected={'cubic':forward_component(a['small_x'],'cubic'),'even':forward_component(canonical,'even'),'even_observed':(a['even_pos']+a['even_neg'])*.5,'large_odd':forward_component(a['large_x']*1000,'odd'),'large_saturation':forward_component(a['large_x']*1000,'sat'),'large_observed':a['large_odd']}
errors={}
for k,v in expected.items():
 assert v.shape==saved[k].shape and np.isfinite(v).all()
 errors[k]=float(np.linalg.norm(v-saved[k])/np.linalg.norm(v));assert errors[k]<1e-12
pairs={'old_rhs_vs_new_rhs':('old_rhs','new_rhs'),'candidate_cubic_vs_old_rhs':('cubic','old_rhs'),'candidate_cubic_vs_new_rhs':('cubic','new_rhs'),'candidate_even_vs_saved_even':('even','even_observed'),'candidate_exact_odd_vs_saved_large_odd':('large_odd','large_observed'),'candidate_saturation_vs_saved_large_odd':('large_saturation','large_observed'),'candidate_saturation_vs_candidate_exact_odd':('large_saturation','large_odd')}
metrics={}
for key,(p,q) in pairs.items():
 x,y=saved[p],saved[q];norm=float(np.linalg.norm(y));assert norm>0
 m={'rows':len(y),'outputs':y.shape[1],'reference_frobenius':norm,'prediction_frobenius':float(np.linalg.norm(x)),'relative_frobenius':float(np.linalg.norm(x-y)/norm),'max_absolute':float(abs(x-y).max())}
 for k,v in m.items():assert np.isclose(v,inner['comparisons'][key][k],rtol=1e-14,atol=0)
 metrics[key]=m
for k in ['target_queries','fits','optimizers','evaluation_reads']:assert inner[k]==0
assert inner['candidate_quality_remains_failed'] and not inner['projection_effect_recomputed'] and not inner['preprojection_coefficients_saved'] and not inner['causal_attribution_proven']
assert inner['reported_unprojected_coupling_tail']==fit['multiscale']['coupling_rank1_max_relative_tail_before_projection']
result={'approved':True,'scope':'independent saved-public-calibration outcome audit only; not candidate acceptance or causal attribution','report_sha256':sha(D/'report.json'),'source_commit':report['source_commit'],'sources_verified':review['sources'],'arrays_sha256':inner['arrays_sha256'],'seven_inputs_verified_unchanged':actual,'independent_einsum_component_relative_discrepancy':errors,'comparisons':metrics,'queries':0,'fits':0,'optimizers':0,'evaluation_arrays_read':0,'preprojection_coefficients_absent':True,'candidate_quality':'FAILED','private_gate':'closed','resources':{'controller_seconds':report['controller_seconds'],'peak_sampled_worker_tree_rss_bytes':report['result']['peak_sampled_rss_bytes'],'samples':report['result']['memory_samples'],'actual_peak_known':False},'limitations':['Calibration estimates are not exact teacher truth.','Residual patterns do not isolate gate error versus coupling or projection effects.','Preprojection arrays were not retained; only logged tail is available.','Two RSS samples cannot establish actual peak or continuous bound.']}
(O/'outcome-review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'report_sha256':result['report_sha256'],'review_sha256':sha(O/'outcome-review.json'),'component_discrepancies':errors},indent=2))
