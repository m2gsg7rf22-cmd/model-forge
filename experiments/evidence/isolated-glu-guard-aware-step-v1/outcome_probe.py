"""Saved-output audit only. No model queries, optimizers or linear solves."""
import json,hashlib,itertools,time,resource,subprocess
from pathlib import Path
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-guard-aware-step-review');D=Path('/Users/adamfaruhi123/.local/share/modelforge-research/experiments/fp32-guard-aware-gate-step-v1-20261025-195d9a6');S=D/'worker/step'
start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
def arr(p):
 with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
maxerr={}
def check(a,b,key,rtol=2e-10,atol=1e-12):
 a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(key,a.shape,b.shape)
 e=float(np.max(np.abs(a-b),initial=0)/max(1.,float(np.max(np.abs(b),initial=0))));maxerr[key]=max(maxerr.get(key,0),e)
 assert np.allclose(a,b,rtol=rtol,atol=atol),(key,e)
def forward(g,c,x):
 t=x@g.T;s=np.empty_like(t);pos=t>=0;s[pos]=1/(1+np.exp(-t[pos]));z=np.exp(t[~pos]);s[~pos]=z/(1+z)
 return np.einsum('nu,nuo->no',t*s,np.einsum('nd,uod->nuo',x,c))
def metrics(e,y):return [float(np.linalg.norm(v)/np.linalg.norm(t)) for v,t in zip(np.split(e,4),np.split(y,4))],[float(np.max(np.abs(v))) for v in np.split(e,4)]
def guards(nr,na,cr,ca,ir,ia):return all(n<=c+1e-10 and n<=i+1e-10 for n,c,i in zip(nr,cr,ir)) and all(n<=c+1e-10*max(1,i) and n<=i+1e-10*max(1,i) for n,c,i in zip(na,ca,ia))
controller=read(D/'controller-report.json');worker=read(D/'worker/worker-report.json');report=read(S/'report.json');approval=read(P/'source-review.json');prereg=R/'experiments/preregistrations/isolated-glu-fp32-guard-aware-gate-step-v1.json';config=read(prereg)
assert sha(D/'worker/worker-report.json')==controller['worker_receipt']['worker_report_sha256'];assert worker==controller['worker_report'];assert worker['inputs_before']==worker['inputs_after']==config['inputs'];assert worker['inputs_unchanged'] is True
assert sha(prereg)==controller['config_sha256'] and sha(P/'source-review.json')==controller['review_sha256']
for name,digest in approval['sources'].items():assert sha(R/name)==digest,name
head=controller['source_commit'];assert head=='195d9a67e0993dbefb5bb6fb3b26a95fe15db3a6'
for name,digest in approval['sources'].items():assert hashlib.sha256(subprocess.check_output(['git','show',head+':'+name],cwd=R)).hexdigest()==digest
for base in [S]:
 manifest=read(base/'artifact-manifest.json');files={str(p.relative_to(base)) for p in base.rglob('*') if p.is_file() and p.name!='artifact-manifest.json'};assert files==set(manifest['files'])
 for name,digest in manifest['files'].items():assert sha(base/name)==digest,name
assert sha(S/'report.json')==worker['step_report_sha256'] and sha(S/'artifact-manifest.json')==worker['artifact_manifest_sha256']
a=arr(S/'initial-problem.npz');g=a['G'];c=a['C'];tx=a['train_x'];ty=a['train_y'];gx=a['guard_x'];gy=a['guard_y'];current=g.copy();row=256
norms=np.array([np.linalg.norm(y) for y in np.split(ty,4)]);w=np.repeat(1/norms,row);weighted=ty*w[:,None]
initial=arr(S/'derived-initial.npz');check(initial['weights'],w,'weights');check(initial['weighted_target'],weighted,'weighted_once')
# Both input probe seeds independently regenerated; no oracle.
for name,seed in [('train',882030),('guard',882031)]:
 rng=np.random.default_rng(seed);xx=np.vstack([rng.normal(size=(256,32))*s for s in [.01,1.,10.,100.]]);assert np.array_equal(a[name+'_x'],xx)
subsets=0;min_geometry=1.;normal_count=0;zero_count=0;scoremap={};directions={};state_snapshots={};maxstation=0.;maxequality=0.;selectedmasks={}
for pid in report['screen']+report['sequential']:
 snap=arr(S/'proposals'/f'{pid}.npz');rec=read(S/'proposals'/f'{pid}.json');u=rec['unit'];gg=snap['current_G'];state_snapshots[pid]=gg
 if pid.startswith('screen'):assert np.array_equal(gg,g)
 # independent complete forward, then audit arithmetic from saved residual.
 check(snap['train_prediction'],forward(gg,c,tx),'proposal_train_forward',atol=2e-7);check(snap['guard_prediction'],forward(gg,c,gx),'proposal_guard_forward',atol=2e-7)
 check(snap['train_residual'],weighted-snap['train_prediction']*w[:,None],'proposal_residual');check(snap['guard_error'],snap['guard_prediction']-gy,'proposal_error')
 t=tx@gg[u];s=.5*(1+np.tanh(t/2));prime=s+t*s*(1-s);branch=tx@c[u].T;res=snap['train_residual'];d=-2*np.einsum('n,no,no,nd->d',w*prime,res,branch,tx);K=np.einsum('n,ni,nj->ij',(w*prime)**2*np.sum(branch**2,axis=1),tx,tx)
 check(rec['gradient'],d,'gradient');check(rec['gram'],K,'gram')
 normals=[];records=rec['normals']['records'];logical=0
 for group in range(4):
  sl=slice(group*row,(group+1)*row);x=gx[sl];y=gy[sl];e=snap['guard_error'][sl];tt=x@gg[u];ss=.5*(1+np.tanh(tt/2));pr=ss+tt*ss*(1-ss);br=x@c[u].T;den=float(np.sum(y*y));M=float(np.max(np.abs(e)));assert M>0
  expected=[('relative_squared',None,None,None,2*np.einsum('n,no,no,nd->d',pr,e,br,x)/den)]
  for j,o in np.argwhere(np.abs(e)>=M*(1-1e-12)):
   sign=1. if e[j,o]>0 else -1.;expected.append(('maximum_absolute',int(j),int(o),sign,sign*pr[j]*br[j,o]*x[j]))
  info=rec['normals']['groups'][group];check(info['reference_squared_norm'],den,'normal_reference');check(info['maximum_absolute'],M,'normal_max')
  for kind,j,o,sign,n in expected:
   rr=records[logical];assert rr['logical_index']==logical and rr['metric']==kind and rr['group']==group
   if j is not None:assert rr['row']==group*row+j and rr['local_row']==j and rr['output']==o and rr['sign']==sign and rr['exact_maximum']==(abs(float(e[j,o]))==M)
   check(rr['normal'],n,'normal',atol=2e-11)
   if np.linalg.norm(n)==0:assert rr['matrix_row'] is None and rr['omitted_reason']=='computed_FP64_zero';zero_count+=1
   else:assert rr['matrix_row']==len(normals);normals.append(np.array(rr['normal']))
   logical+=1
 assert logical==len(records)==8;normal_count+=logical
 A=np.vstack(normals);problem=arr(S/'cones/proposals'/pid/'problem.npz');check(problem['gradient'],d,'cone_d');check(problem['gram'],K,'cone_K');check(problem['constraints'],A,'cone_A');check(problem['gate'],gg[u],'cone_gate')
 tr=read(S/'cones/proposals'/pid/'trace.json');out=tr['result'];K=problem['gram'];d=problem['gradient'];eig=np.linalg.eigvalsh(K);beta=float(eig[-1]);assert eig[0]>1e-12*beta;kn=K/beta;dn=d/beta;B=A/np.linalg.norm(A,axis=1)[:,None];best=None
 assert len(tr['subsets'])==2**len(A)==tr['linear_solves_attempted']
 for mask,rr in enumerate(tr['subsets']):
  active=[i for i in range(len(A)) if mask>>i&1];assert rr['mask']==mask and rr['active']==active
  if active:
   sv=np.linalg.svd(B[active],compute_uv=False);check(rr['singular_values'],sv,'geometry_singular');ratio=float(sv[-1]/sv[0]);assert ratio>1e-5;min_geometry=min(min_geometry,ratio);check(rr['singular_ratio'],ratio,'geometry_ratio')
  z=np.array(rr['solution']);mu=np.array(rr['normalized_duals']);assert np.isfinite(z).all() and np.isfinite(mu).all();assert all(mu[i]==0 for i in range(len(A)) if i not in active)
  force=2*kn@z;normal=B.T@mu;station=dn+force+normal;az=B@z;comp=mu*az
  maxstation=max(maxstation,float(np.max(np.abs(station))));maxequality=max(maxequality,float(np.max(np.abs(az[active]),initial=0)))
  assert np.max(np.abs(station))<1e-12 and np.max(np.abs(az[active]),initial=0)<1e-12
  checks={'primal':float(np.max(np.maximum(0,az)/np.maximum(1,np.abs(az)),initial=0)),'dual':float(np.max(np.maximum(0,-mu)/np.maximum(1,np.abs(mu)),initial=0)),'stationarity':float(np.max(np.abs(station)/np.maximum.reduce([np.ones_like(d),abs(dn),abs(force),abs(normal)]),initial=0)),'complementarity':float(np.max(abs(comp)/np.maximum.reduce([np.ones_like(mu),abs(mu),abs(az)]),initial=0))}
  for key,value in checks.items():check(rr['checks'][key],value,'kkt_'+key,atol=1e-14)
  obj=float(dn@z+z@kn@z);check(rr['objective'],obj,'subset_objective');passed=all(v<=1e-10 for v in checks.values());assert rr['status']==('numerical_kkt_passed' if passed else 'numerical_kkt_rejected')
  if passed and (best is None or obj<best[0]):best=(obj,mask,z,mu)
  subsets+=1
 assert best is not None and out['selected_mask']==best[1];check(out['unclipped'],best[2],'selected_raw');factor=min(1.,.05*np.linalg.norm(gg[u])/np.linalg.norm(best[2])) if np.linalg.norm(best[2]) else 1.;delta=best[2]*factor;check(rec['direction'],delta,'clipped_direction');check(out['original_duals'],beta*best[3]/np.linalg.norm(A,axis=1),'original_duals');score=-float(d@delta+delta@K@delta);check(rec['score'],score,'score');scoremap[pid]=rec['score'];directions[pid]=np.array(rec['direction']);selectedmasks[pid]=best[1]
 assert time.monotonic()-start<165,'audit exceeded bounded budget'
# Stable screen selection uses saved scores after their independent arithmetic checks.
scores=[scoremap[p] for p in report['screen']];order=sorted(range(64),key=lambda i:(-scores[i],i));chosen=[i for i in order if scores[i]>max(1e-12*report['initial_cost'],1e-24)][:8];assert order==report['ordered'] and chosen==report['selected']
ir,ia=metrics(initial['guard_error'],gy);check(ir,report['initial_guard_relative'],'initial_guard_rel');check(ia,report['initial_guard_max_absolute'],'initial_guard_abs');accepted=[];cursor=0
for u in chosen:
 pid='sequential-'+str(u);assert np.array_equal(state_snapshots[pid],current);beforepred=forward(current,c,tx);beforeguard=forward(current,c,gx);cr,ca=metrics(beforeguard-gy,gy);cost0=float(np.sum((weighted-beforepred*w[:,None])**2));old=current[u].copy();delta=directions[pid]
 for power in range(8):
  trial=report['trials'][cursor];assert trial['unit']==u and trial['alpha']==2.**-power;expected=current.copy();expected[u]=old+2.**-power*delta;check(trial['gate'],expected[u],'trial_gate')
  data=arr(S/'trials'/f'{cursor}.npz');raw=arr(S/'trials'/f'{cursor}-predictions.npz');assert np.array_equal(data['G'],expected);fp=forward(expected,c,tx);fg=forward(expected,c,gx);check(data['train_prediction'],fp,'trial_train_forward',atol=2e-7);check(data['guard_prediction'],fg,'trial_guard_forward',atol=2e-7)
  for tag in ['train','guard']:assert np.allclose(raw['cached_'+tag+'_prediction'],raw['full_'+tag+'_prediction'],rtol=1e-12,atol=1e-12*max(1,float(np.max(np.abs(raw['full_'+tag+'_prediction'])))))
  nr=weighted-data['train_prediction']*w[:,None];ne=data['guard_prediction']-gy;check(data['train_residual'],nr,'trial_residual');check(data['guard_error'],ne,'trial_error');cost1=float(np.sum(nr**2));ar,aa=metrics(ne,gy);check(cost1,trial['proposed_cost'],'trial_cost');check(cost0,trial['old_cost'],'old_cost');check(ar,trial['guard_after_relative'],'trial_rel');check(aa,trial['guard_after_max_absolute'],'trial_abs');check(cr,trial['guard_before_relative'],'before_rel');check(ca,trial['guard_before_max_absolute'],'before_abs')
  dec=cost1<cost0-max(1e-12*cost0,1e-24);ok=guards(ar,aa,cr,ca,ir,ia);assert trial['training_decreased']==dec and trial['guard_passed']==ok and trial['accepted']==(dec and ok);cursor+=1
  if dec and ok:current=expected;accepted.append(u);break
assert cursor==len(report['trials'])==9 and accepted==report['accepted'];assert len(directions)==72 and subsets==14336
candidate=arr(S/'candidate.npz');assert np.array_equal(candidate['gates'],current) and candidate['couplings'].tobytes()==c.tobytes();singular=np.linalg.svd(c,compute_uv=False);assert (singular[:,0]>0).all() and (singular[:,1]<=1e-10*singular[:,0]).all();assert sha(S/'candidate.npz')==worker['candidate_sha256']=='7f7d69552fe3ad8c27ad17b5f8326038fd7d26a2171ad84d8660d07560aa0a06'
final=arr(S/'evidence/numeric-evidence.npz');check(final['final_train_residual'],weighted-forward(current,c,tx)*w[:,None],'final_residual',atol=1e-12);check(final['final_guard_error'],forward(current,c,gx)-gy,'final_error',atol=2e-7);cost=float(np.sum(final['final_train_residual']**2));fr,fa=metrics(final['final_guard_error'],gy);check(cost,report['final_cost'],'final_cost');check(fr,report['final_guard_relative'],'final_rel');check(fa,report['final_guard_max_absolute'],'final_abs');assert guards(fr,fa,ir,ia,ir,ia)
for receipt in [controller['preflight'],controller['worker_receipt']]:assert receipt['process_monitor_scope']=='process_and_children' and receipt['memory_samples']>0 and receipt['peak_sampled_rss_bytes']<512*1024**2
result={'approved':True,'scope':'saved actual-fit evidence and adaptive calibration improvement only; NO quality/evaluation claim','report_sha256':sha(D/'controller-report.json'),'candidate_sha256':sha(S/'candidate.npz'),'source_commit':head,'sources_verified':len(approval['sources']),'six_inputs_before_after_match_frozen':True,'all_artifact_hashes_verified':True,'proposals':72,'saved_solutions_checked_without_solve':subsets,'normal_records':normal_count,'zero_omissions':zero_count,'min_independent_geometry_ratio':min_geometry,'max_absolute_stationarity_equation_residual':maxstation,'max_absolute_active_equality_residual':maxequality,'selected':chosen,'accepted':accepted,'trials':cursor,'initial_train_cost':report['initial_cost'],'final_train_cost':cost,'relative_cost_reduction':1-cost/report['initial_cost'],'all4_guard_relative_improved':all(f<i for f,i in zip(fr,ir)),'all4_guard_maxabs_improved':all(f<i for f,i in zip(fa,ia)),'final_relative_guard':fr,'final_maxabs_guard':fa,'worker_peak_sampled_tree_rss_bytes':controller['worker_receipt']['peak_sampled_rss_bytes'],'worker_memory_samples':controller['worker_receipt']['memory_samples'],'preflight_scope':'process_and_children','actual_queries':0,'new_fits':0,'new_direction_solves':0,'consumed_validation_reads':0,'quality_evaluation_authorized':False,'private_gate':'closed','max_normalized_comparison_errors':maxerr,'audit_elapsed_seconds':time.monotonic()-start,'audit_self_maxrss_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limitations':['RSS polled samples are not continuous peak proof or parent-controller RSS','Current/initial guard data are adaptive reused calibration, not independent validation','941033 remains unreserved and unused; separate prospective review required','Frozen input before/after commitments checked from receipts and saved approved inputs; old full report content never decoded','Prior quality failures and original failed resource receipts remain unchanged']}
(P/'outcome-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['max_normalized_comparison_errors','limitations']}))
