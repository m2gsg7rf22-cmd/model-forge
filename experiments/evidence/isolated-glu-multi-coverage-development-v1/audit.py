from pathlib import Path
import ast,hashlib,json,time
import numpy as np
repo=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository'); run=Path('/Users/adamfaruhi123/.local/share/modelforge-research/experiments/isolated-glu-multi-coverage-development-v1-20261025'); inp=Path('/Users/adamfaruhi123/.local/share/modelforge-research/experiments/isolated-glu-fullmap-v3-20261025-a0d0556/public'); out=Path('/private/tmp/modelforge-multi-independent-review')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
source=repo/'examples/isolated_glu/coverage_multi.py'; plan=read(run/'plan.json'); summary=read(run/'summary.json');assert plan['source_sha256']==sha(source)
def fns(path):return {f.name:ast.dump(f,include_attributes=False) for f in ast.parse(path.read_text()).body if isinstance(f,ast.FunctionDef)}
a,b=fns(source),fns(repo/'examples/isolated_glu/coverage.py'); helper_match={k:a[k]==b[k] for k in ['pack','unpack','tangent','residual_subspace','refine']};assert all(helper_match.values())
i,j=np.triu_indices(32);w=np.where(i==j,1.,np.sqrt(2.))
def pack(a):return a[...,i,j]*w
def unpack(v):
 a=np.zeros((32,32));a[i,j]=v/w;a[j,i]=a[i,j];return a
def loss(d,basis):
 q=d[:,i]*d[:,j]*w;r=q-(q@basis.T)@basis;return np.sum(r*r,axis=1)
def components(d,basis):
 rows=[]
 for x in d:
  t=np.stack([pack(np.outer(x,z)+np.outer(z,x)) for z in np.eye(32)],axis=1);t=np.linalg.qr(t,mode='reduced')[0];_,_,vt=np.linalg.svd(t-basis.T@(basis@t),full_matrices=False);rows.extend([basis@(t@vt[-1]),basis@(t@vt[-2])])
 _,s,vt=np.linalg.svd(np.stack(rows),full_matrices=True);rank=int((s>1e-8*s[0]).sum());return vt[rank:]@basis,rank
rows=[]; started=time.perf_counter()
for k in range(3):
 p=run/str(k); original=inp/str(k); report=read(p/'report.json'); monitored=read(run/f'monitored-{k}.json'); srow=summary['results'][k]
 assert report['source_sha256']==sha(source) and report['input_count']==plan['input_counts'][k]; assert report['input_sha256']=={n:sha(original/n) for n in ['hessians.npz','refined-solution.npz']}; assert all(monitored[n]==v for n,v in report.items());assert monitored['index']==k
 assert monitored['process_monitor_scope']=='process_and_children' and monitored['memory_limit_mb']==512 and monitored['memory_samples']>0 and 0<monitored['peak_sampled_rss_bytes']<=512*1024**2 and 0<=monitored['seconds']<180
 assert all(srow[n]==monitored[n] for n in ['input_count','retained_count','seconds','process_monitor_scope','peak_sampled_rss_bytes','memory_samples'])
 h=load(original/'hessians.npz')['hessian'];known=load(original/'refined-solution.npz')['directions']; _,sv,vt=np.linalg.svd(pack(h).reshape(-1,528),full_matrices=False); basis=vt[:128]; assert sv[127]>=100*sv[128] and sv[127]>1e-10*sv[0]
 rng=np.random.default_rng(20261025);rounds=[]
 for n,r in enumerate(report['rounds']):
  assert n<4 and r['round']==n and 'stop' not in r; file=p/f'round-{n}.npz';assert r['artifact_sha256']==sha(file);z=load(file);assert set(z)=={'input_directions','starts','all_directions','residual','directions'};assert np.array_equal(z['input_directions'],known);assert z['starts'].shape==z['all_directions'].shape==(512,32) and z['residual'].shape==(512,)
  assert all(v.dtype==np.float64 and np.isfinite(v).all() for v in z.values());assert np.allclose(np.linalg.norm(z['starts'],axis=1),1,rtol=0,atol=1e-10);assert np.allclose(np.linalg.norm(z['all_directions'],axis=1),1,rtol=0,atol=1e-10);assert (z['residual']>=0).all()
  after=loss(z['all_directions'],basis);before=loss(z['starts'],basis);assert np.allclose(after,z['residual'],rtol=1e-6,atol=1e-12);assert np.max(after-before)<=1e-12
  residual,rank=components(known,basis);assert rank==2*len(known)==r['geometry']['component_rank'];assert len(residual)==128-2*len(known)==r['geometry']['residual_dimension'];starts=[]
  for _ in range(256):
   c=rng.normal(size=len(residual));c/=np.linalg.norm(c);eig,v=np.linalg.eigh(unpack(c@residual));starts.extend(v[:,idx] for idx in np.argsort(np.abs(eig))[-2:])
  start_cos=np.abs(np.sum(np.stack(starts)*z['starts'],axis=1));assert start_cos.min()>1-1e-10
  selected=list(known)
  for idx in np.argsort(z['residual']):
   if z['residual'][idx]<=1e-8 and all(abs(z['all_directions'][idx]@d)<.9995 for d in selected):selected.append(z['all_directions'][idx])
  expected=np.stack(selected);assert np.array_equal(expected,z['directions']);assert len(known)==r['input_count'] and len(expected)==r['retained_count'];known=expected
  rounds.append({'round':n,'input_count':r['input_count'],'retained_count':len(known),'component_rank':rank,'residual_dimension':len(residual),'all512_loss_consistent':True,'initialization_min_absolute_cosine':float(start_cos.min()),'max_after_minus_before_squared_loss':float(np.max(after-before)),'selection_exact_no_truncation':True,'artifact_sha256':sha(file)})
 final=load(p/'multi-coverage-solution.npz')['directions'];assert np.array_equal(final,known) and len(final)==report['retained_count']==64;cos=np.abs(final@final.T);np.fill_diagonal(cos,0);assert cos.max()<.9995;final_loss=loss(final,basis);assert final_loss.max()<=1e-8;assert np.allclose(np.linalg.norm(final,axis=1),1,rtol=0,atol=1e-10)
 rows.append({'index':k,'input_count':report['input_count'],'retained_count':len(final),'rounds':rounds,'max_retained_original_subspace_squared_loss':float(final_loss.max()),'max_pairwise_absolute_cosine':float(cos.max()),'prefix_unchanged':True,'source_input_round_hashes_match':True,'final_solution_sha256':sha(p/'multi-coverage-solution.npz'),'seconds':monitored['seconds'],'peak_sampled_rss_bytes':monitored['peak_sampled_rss_bytes']})
result={'status':'verified_calibration_only_geometry','source_sha256':sha(source),'plan_sha256':sha(run/'plan.json'),'summary_sha256':sha(run/'summary.json'),'copied_geometric_helpers_ast_identical':helper_match,'results':rows,'audit_seconds':time.perf_counter()-started,'no_fitting_or_queries':True,'no_private_teacher_or_heldout_access':True,'teacher_identity_verified':False,'fullmap_verified':False,'formal_v3_outcome_changed':False,'private_final_model_gate':'closed','blocking_findings':[],'limits':['Postfailure calibration-only development using old publicH, not fresh replication.','64 candidates, rank and tiny residual do not prove all true gates or fullmap fidelity.','Checks replay initialization geometry and greedy selection but not GN fitting; iteration count and algorithm supported by frozen source.','Synthetic controls and these3development inputs do not establish generality.','search returns extras or stalled counts; any fresh protocol must explicitly require exactly64 and reject65plus.','Runtime uses process-tree sampledRSS, not reserved or continuouslymeasured peak.']}
(out/'outcome-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));(out/'outcome-audit.md').write_text('# ביקורת כיול ציבורי — כיסוי מרובה\n\nאומתו60→64,62→64,61→64בסיבוב אחד לכל מערך, עם ממדי שארית8/4/6. המקור והקלט תואמים להתחייבויות;512האתחולים בכל סיבוב שוחזרו גאומטרית, ללא הפעלת fitting. הפסדי כל המועמדים חושבו מולVהמקורי וכלל הבחירה שוחזר ללאtruncation. הקידומת נשמרה ואין כפילויות.\n\nהביקורת לא קראה אמת מורה אוheldoutולא ביצעה שאילתות. זהות השערים ודיוק הפונקציה לא אומתו. כישלוןv3נשאר; v4טרם נבדק; השער הפרטי סגור. פירוט ב־outcome-audit.json.\n')
