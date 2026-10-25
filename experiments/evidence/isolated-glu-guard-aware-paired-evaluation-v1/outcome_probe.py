"""Audit saved paired941033outputs only; no oracle import, query or fitting."""
import json,hashlib,subprocess,time,resource
from pathlib import Path
import numpy as np
P=Path('/private/tmp/modelforge-guard-aware-evaluation-review');R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');B=Path('/Users/adamfaruhi123/.local/share/modelforge-research/experiments');D=B/'fp32-guard-aware-paired-eval-v1-20261025-9357159';F=B/'fp32-guard-aware-gate-step-v1-20261025-195d9a6';W=D/'worker';start=time.monotonic()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
def arrays(path):
 with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def independent_forward(g,c,x):
 t=x@g.T;s=np.empty_like(t);positive=t>=0;s[positive]=1/(1+np.exp(-t[positive]));exp=np.exp(t[~positive]);s[~positive]=exp/(1+exp)
 return np.einsum('nu,nuo->no',t*s,np.einsum('nd,uod->nuo',x,c))
controller=read(D/'controller-report.json');worker=read(W/'worker-report.json');pair=read(W/'batches/progress.json');approval=read(P/'source-review.json');prereg=R/'experiments/preregistrations/isolated-glu-fp32-guard-aware-paired-evaluation-v1.json';config=read(prereg)
assert sha(D/'controller-report.json')=='f9f652f42abcfb05ad49bf43f7265f2ba5ae73ec04eb1a8d9809716bd9840275';assert controller['status']=='completed' and controller['evaluation_completed'] and controller['quality_criterion_met'] is False
assert sha(prereg)==controller['config_sha256'] and sha(P/'source-review.json')==controller['review_sha256'];head=controller['source_commit'];assert head=='9357159266033fc23dfd3e4f8e7bd0a143bea104'
for name,digest in approval['sources'].items():
 assert sha(R/name)==digest
 assert hashlib.sha256(subprocess.check_output(['git','show',head+':'+name],cwd=R)).hexdigest()==digest
subprocess.run(['git','merge-base','--is-ancestor',head,'HEAD'],cwd=R,check=True)
paths={'candidate':F/'worker/step/candidate.npz','baseline':B/'fp32-guarded-gate-step-v1-20261025-3d3b622/candidate/candidate.npz','fit_controller':F/'controller-report.json','fit_worker':F/'worker/worker-report.json','fit_step':F/'worker/step/report.json','fit_artifact_manifest':F/'worker/step/artifact-manifest.json','outcome_audit':R/'experiments/evidence/isolated-glu-guard-aware-step-v1/outcome-audit.json'}
actual={k:sha(p) for k,p in paths.items()};assert actual==config['inputs']==worker['inputs_before']==worker['inputs_after'];assert worker['inputs_unchanged'] is True
assert controller['worker_report']==worker and worker['evaluation']==pair;assert sha(W/'worker-report.json')==controller['worker_receipt']['worker_report_sha256'];assert sha(W/'batches/progress.json')==worker['paired_report_sha256']
files={str(p.relative_to(W)) for p in W.rglob('*') if p.is_file()};assert files==set(controller['worker_files_sha256'])
for name,digest in controller['worker_files_sha256'].items():assert sha(W/name)==digest
marker=F/'public941033-evaluation-attempt.json';assert str(marker)==worker['persistent_marker'] and sha(marker)==worker['persistent_marker_sha256'];reservation=read(marker)
assert reservation=={'calls_reserved':2048,'candidate_sha256':actual['candidate'],'baseline_sha256':actual['baseline'],'config_content_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),'evaluation_seed':941033,'automatic_retry':False}
assert worker['reservation_status']=='metadata_persisted'
for record in [controller['worker_receipt'],worker,pair]:
 for k in ['calls_reserved','calls_attempted','calls_returned','calls_persisted']:assert record[k]==2048
# Entire previous fit root unchanged except the exactly named new marker.
fitmanifest=read(paths['fit_artifact_manifest']);expected={'worker/step/'+name for name in fitmanifest['files']};expected.update(['worker/step/artifact-manifest.json','worker/worker-report.json','controller-report.json','public941033-evaluation-attempt.json'])
assert {str(p.relative_to(F)) for p in F.rglob('*') if p.is_file()}==expected
for name,digest in fitmanifest['files'].items():assert sha(F/'worker/step'/name)==digest
student=arrays(paths['candidate']);baseline=arrays(paths['baseline']);assert student['couplings'].tobytes()==baseline['couplings'].tobytes() and student['gates'].tobytes()!=baseline['gates'].tobytes()
raw=arrays(W/'batches/paired-validation.npz');assert set(raw)=={'inputs','reference','prediction','baseline_prediction'}
for value in raw.values():assert value.shape==(2048,32) and value.dtype==np.float64 and np.isfinite(value).all()
assert sha(W/'batches/paired-validation.npz')==pair['validation_sha256']=='e7c3350cc40b287afcb7ef007ea341366b09010de83bc68bd3ec65cac5c88458'
rng=np.random.default_rng(941033);expected_x=np.vstack([rng.normal(size=(512,32))*s for s in [.01,1.,10.,100.]]);assert expected_x.tobytes()==raw['inputs'].tobytes()
for group in range(4):
 sl=slice(group*512,(group+1)*512)
 for suffix,key,filekey in [('inputs','inputs','inputs'),('reference','reference','reference'),('student','prediction','prediction'),('baseline','baseline_prediction','prediction')]:
  a=arrays(W/f'batches/group-{group}-{suffix}.npz');assert set(a)=={filekey};assert a[filekey].tobytes()==raw[key][sl].tobytes()
assert raw['reference'].astype(np.float32).astype(np.float64).tobytes()==raw['reference'].tobytes()
errors={}
for label,model,key in [('candidate',student,'prediction'),('baseline',baseline,'baseline_prediction')]:
 pred=independent_forward(model['gates'],model['couplings'],raw['inputs']);errors[label]=float(np.linalg.norm(pred-raw[key])/np.linalg.norm(raw[key]));assert errors[label]<1e-12
computed=[]
for group,scale in enumerate([.01,1.,10.,100.]):
 sl=slice(group*512,(group+1)*512);truth=raw['reference'][sl];pred=raw['prediction'][sl];base=raw['baseline_prediction'][sl];delta=pred-truth;bd=base-truth;rel=float(np.linalg.norm(delta)/np.linalg.norm(truth));ab=float(np.max(np.abs(delta)));br=float(np.linalg.norm(bd)/np.linalg.norm(truth));ba=float(np.max(np.abs(bd)))
 row={'scale':scale,'relative_frobenius':rel,'max_absolute':ab,'baseline_relative_frobenius':br,'baseline_max_absolute':ba,'relative_difference':rel-br,'max_absolute_difference':ab-ba,'relative_regressed':rel>br,'max_absolute_regressed':ab>ba}
 assert row==pair['groups'][group];assert rel>.001 and rel<br and ab<ba;computed.append(row)
assert pair['all_groups_relative_at_most_1e_minus3'] is False and pair['development_max_group_relative']==.001
assert controller['elapsed_seconds_before_final_report']<180
for receipt in [controller['preflight'],controller['worker_receipt']]:assert receipt['process_monitor_scope']=='process_and_children' and receipt['memory_samples']>=1 and receipt['peak_sampled_rss_bytes']<512*1024**2
result={'approved':True,'scope':'saved paired PUBLIC941033evaluation evidence VALID; quality criterion FAILED; no promotion','report_sha256':sha(D/'controller-report.json'),'validation_sha256':sha(W/'batches/paired-validation.npz'),'source_commit':head,'source_files_verified':len(approval['sources']),'all7_original_input_files_rehashed':actual,'inputs_before_after_exact':True,'fitroot_unchanged_except_exact_marker':True,'all_worker_files_inventory_hashes_verified':True,'persistent_marker_sha256':sha(marker),'seed_status':'941033_CONSUMED_NO_RETRY','reserved_attempted_returned_persisted':[2048]*4,'all4_groups_raw_aggregate_byteidentical':True,'all2048_rng_points_byteidentical':True,'reference_fp32_values_stored_exact_fp64':True,'independent_sigmoid_einsum_relative_errors':errors,'groups':computed,'all4_relative_improve':True,'all4_maxabs_improve':True,'all4_relative_FAIL_fixed_0_001':True,'quality_criterion_met':False,'worker_peak_sampled_tree_rss_bytes':controller['worker_receipt']['peak_sampled_rss_bytes'],'worker_memory_samples':controller['worker_receipt']['memory_samples'],'preflight_peak_sampled_tree_rss_bytes':controller['preflight']['peak_sampled_rss_bytes'],'preflight_memory_samples':controller['preflight']['memory_samples'],'elapsed_seconds_before_final_report':controller['elapsed_seconds_before_final_report'],'new_reference_queries_in_audit':0,'new_fits':0,'older_consumed_validation_reads':0,'private_gate':'closed','audit_probe_sha256':sha(Path(__file__)),'audit_elapsed_seconds':time.monotonic()-start,'audit_maxrss_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limitations':['QualityFAIL despite allpaired relative and absolute improvements; no criterion relaxation or extra evaluation','Owned public fixture informed by prior development, not blind hidden-teacher/fullTransformer/universal equivalence','Sampled worker-tree RSS does not prove continuous peak or parent-controller memory','elapsed_seconds_before_final_report excludes final report write; approximate wall-time evidence only','Marker and terminal consumed status prohibit retry; counts are saved execution evidence not a crash-proof transactional oracle','Prior quality failures and monitor failures preserved; final private mission gate remains closed']}
(P/'outcome-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['groups','all7_original_input_files_rehashed','limitations']}))
