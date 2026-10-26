"""Saved-array audit ONLY: never import registered_contexts, forward or oracle code."""
import hashlib,json,re,subprocess,time,zipfile,gc
from pathlib import Path
import numpy as np
P=Path('/private/tmp/modelforge-dimension-domain-outcome-review');R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');D=Path('/Users/adamfaruhi123/.local/share/modelforge-research/experiments/dimension-domain-v1-20261026-01b45d9');W=D/'worker';A=W/'arrays';start=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024**2):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def load(name,shapes):
 p=A/name
 with zipfile.ZipFile(p) as z:assert sum(i.file_size for i in z.infolist())<=8*1024**2
 with np.load(p,allow_pickle=False) as z:
  assert set(z.files)==set(shapes),name
  values={k:z[k] for k in z.files}
 for key,v in values.items():assert v.shape==shapes[key] and v.dtype==np.float64 and np.isfinite(v).all(),(name,key)
 return values
errors={}
def close(a,b,key,rtol=3e-10,atol=2e-12):
 assert np.shape(a)==np.shape(b),key
 aa=np.asarray(a);bb=np.asarray(b);er=float(np.max(np.abs(aa-bb),initial=0)/max(1.,float(np.max(np.abs(bb),initial=0))));errors[key]=max(errors.get(key,0),er)
 assert np.allclose(aa,bb,rtol=rtol,atol=atol),(key,er)
def atomvec(H,n):
 return np.stack([H[...,i,j]*(1. if i==j else np.sqrt(2.)) for i in range(n) for j in range(i,n)],axis=-1)
def stable_norm(a):
 s=float(np.max(np.abs(a),initial=0));return 0. if s==0 else float(s*np.sqrt(np.sum((a/s)**2)))
def residuals(projector,v):
 v=v/np.max(abs(v),axis=1)[:,None];v=v/np.sqrt(np.sum(v*v,axis=1))[:,None];atom=atomvec(v[:,:,None]*v[:,None,:],v.shape[1]);rr=atom-atom@projector
 scale=np.max(abs(rr),axis=1);div=np.where(scale==0,1.,scale);return scale*np.sqrt(np.sum((rr/div[:,None])**2,axis=1))
controller=read(D/'controller-report.json');report=read(W/'worker-report.json');progress=read(W/'progress.json');approval=read(Path('/private/tmp/modelforge-domain-run-review/source-review.json'));configpath=R/'experiments/preregistrations/isolated-glu-dimension-domain-v1.json';config=read(configpath)
assert sha(D/'controller-report.json')=='ec13f57abb5253522be6d9ea591392578c03a59dbeb62d711eea40a71e764dcf'
assert controller['status']=='completed' and controller['diagnostic_completed'] is True and controller['worker_report']==report
assert sha(configpath)==controller['config_sha256'] and sha(Path('/private/tmp/modelforge-domain-run-review/source-review.json'))==controller['review_sha256']
head=controller['source_commit'];assert head=='01b45d9e112781ef39b61f710e7b9c6af39cdf3a'
for name,digest in approval['sources'].items():
 assert sha(R/name)==digest,name
 assert hashlib.sha256(subprocess.check_output(['git','show',head+':'+name],cwd=R)).hexdigest()==digest,name
subprocess.run(['git','merge-base','--is-ancestor',head,'HEAD'],cwd=R,check=True)
assert sha(W/'worker-report.json')==controller['worker_receipt']['worker_report_sha256']
for item in [controller['worker_receipt'],report,progress]:
 for key in ['calls_reserved','calls_attempted','calls_returned','calls_persisted']:assert item[key]==1454080
marker=Path(report['marker']);assert marker==R.parent/'experiments/isolated-glu-dimension-domain-v1-attempt.json' and sha(marker)==report['marker_sha256'];reservation=read(marker)
assert reservation=={'policy':config['policy'],'calls_reserved':1454080,'source_commit':head,'config_sha256':sha(configpath),'output':str(W.absolute()),'automatic_retry':False}
# Independent exact fixed names and streaming commitments.
names=set()
for profile in range(4):
 for teacher in range(2):
  for channel in ['fp64','fp32']:
   case=f'p{profile}-t{teacher}-{channel}';names.add(case+'-context.npz')
   for step in range(2):
    for row in range(64):
     for kind in ['inputs','responses']:names.add(f'{case}-s{step}-x{row:02d}-{kind}.npz')
   for kind in ['reference','fine','coarse','richardson']:
    for chunk in range(8):names.add(f'{case}-{kind}-h{chunk:02d}.npz')
   for kind in ['atoms','reference','fine','coarse','richardson']:names.add(f'{case}-spectrum-{kind}.npz')
assert len(names)==4704 and {p.name for p in A.iterdir()}==names|{'manifest.jsonl'}
manifest=A/'manifest.jsonl';assert sha(manifest)==report['artifact_manifest_sha256'];seen=set();total=manifest.stat().st_size;largest=0
with manifest.open('rb') as f:
 for line in f:
  assert line.endswith(b'\n') and len(line)<1024;item=json.loads(line);name=item['name'];assert set(item)=={'name','bytes','sha256'} and name in names and name not in seen;seen.add(name);p=A/name
  assert p.is_file() and not p.is_symlink() and p.stat().st_size==item['bytes'] and 0<item['bytes']<=8*1024**2 and sha(p)==item['sha256'],name
  total+=item['bytes'];largest=max(largest,item['bytes'])
assert seen==names and report['artifact_inventory']=={'shards':4704,'bytes_reserved':total,'largest_shard_bytes':largest}
# Verify retained synthetic transport commitments, not regenerate any data.
T=D/'transport';transport_total=(T/'manifest.jsonl').stat().st_size;transport_max=0;transportnames=set()
for line in (T/'manifest.jsonl').read_text().splitlines():
 item=json.loads(line);transportnames.add(item['name']);assert sha(T/item['name'])==item['sha256'] and (T/item['name']).stat().st_size==item['bytes'];transport_total+=item['bytes'];transport_max=max(transport_max,item['bytes'])
assert {p.name for p in T.iterdir()}==transportnames|{'manifest.jsonl'} and len(transportnames)==3
assert transport_total==controller['transport_preflight']['bytes_reserved'] and transport_max==controller['transport_preflight']['largest_shard_bytes']
profiles=[(4,8,4),(8,16,8),(16,32,16),(32,64,32)];case_results=[];counts=0;batch_count=0;raw_response_values=0;analytic_max=0.;rng_bindings=0
assert len(report['cases'])==16
for index,stored in enumerate(report['cases']):
 j=index//4;teacher=(index%4)//2;channel='fp64' if index%2==0 else 'fp32';n,m,o=profiles[j];p=n*(n+1)//2;case=f'p{j}-t{teacher}-{channel}';assert stored['case']==case and stored['channel']==('fp64' if channel=='fp64' else 'fp32_compute')
 shapes={'g':(m,n),'u':(m,n),'d':(o,m),'reference_g':(m,n),'reference_u':(m,n),'reference_d':(o,m),'probes':(64,n),'random_directions':(64,n),'atoms':(2*m,p),'preactivations':(64,m)};ctx=load(case+'-context.npz',shapes)
 # Allowed read-only RNG bindings of SAVED public arrays; never evaluate their model.
 rng=np.random.default_rng(684031+100*j+teacher)
 for key,shape in [('g',(m,n)),('u',(m,n)),('d',(o,m))]:assert ctx[key].tobytes()==(rng.normal(size=shape)*.2).tobytes();rng_bindings+=1
 xp=np.random.default_rng(684033+100*j).normal(size=(64,n))*4;rd=np.random.default_rng(684034+100*j).normal(size=(64,n));assert xp.tobytes()==ctx['probes'].tobytes() and rd.tobytes()==ctx['random_directions'].tobytes();rng_bindings+=2
 for key in ['g','u','d']:
  reference=ctx[key] if channel=='fp64' else ctx[key].astype(np.float32).astype(np.float64);assert reference.tobytes()==ctx['reference_'+key].tobytes()
 g,u,d=(ctx['reference_'+key] for key in ['g','u','d']);z=xp@g.T;uv=xp@u.T;close(ctx['preactivations'],z,'preactivations')
 for key,actual in [('preactivation_mean',float(z.mean())),('preactivation_std',float(z.std())),('preactivation_min',float(z.min())),('preactivation_max',float(z.max()))]:close(stored[key],actual,'preactivation_statistics')
 gg=g[:,:,None]*g[:,None,:];gu=g[:,:,None]*u[:,None,:];cross=gu+gu.swapaxes(-1,-2);atoms=atomvec(np.stack([gg,cross],axis=1),n).reshape(2*m,p);assert np.array_equal(ctx['atoms'],atoms)
 def chunks(kind):
  return np.concatenate([load(f'{case}-{kind}-h{k:02d}.npz',{'hessian':(8,o,n,n)})['hessian'] for k in range(8)],axis=0)
 href=chunks('reference');assert np.array_equal(href,href.swapaxes(-1,-2))
 # Independent analytic product rule; stable derivative with exp(-abs(z)), no forward oracle.
 sig=np.empty_like(z);positive=z>=0;sig[positive]=1/(1+np.exp(-z[positive]));ez=np.exp(z[~positive]);sig[~positive]=ez/(1+ez);tail=np.exp(-np.abs(z));sp=tail/(1+tail)**2;first=sig+z*sp;second=sp*(2+z*(1-2*sig))
 analytic_sq=reference_sq=0.
 for k in range(64):
  H=np.einsum('om,mij->oij',d*(second[k]*uv[k])[None,:],gg)+np.einsum('om,mij->oij',d*first[k][None,:],cross)
  close(href[k],H,'analytic_product_rule',atol=3e-12);analytic_max=max(analytic_max,float(np.max(abs(H-href[k]))));analytic_sq+=float(np.sum((H-href[k])**2));reference_sq+=float(np.sum(href[k]**2))
 hvalues={'reference':href};pairs=[(i,jj) for i in range(n) for jj in range(i,n)];ii=np.array([v[0] for v in pairs]);jj=np.array([v[1] for v in pairs]);eye=np.eye(n)
 for sid,h in enumerate([.3,.6]):
  computed=np.empty((64,o,n,n))
  for row,x in enumerate(xp):
   xx=load(f'{case}-s{sid}-x{row:02d}-inputs.npz',{'inputs':(4*p,n)})['inputs'];yy=load(f'{case}-s{sid}-x{row:02d}-responses.npz',{'responses':(4*p,o)})['responses'];ei=eye[ii]*h;ej=eye[jj]*h;q=np.stack([x+ei+ej,x+ei-ej,x-ei+ej,x-ei-ej]);assert xx.tobytes()==q.reshape(4*p,n).tobytes(),(case,sid,row)
   if channel=='fp32':assert yy.astype(np.float32).astype(np.float64).tobytes()==yy.tobytes()
   counts+=4*p;batch_count+=1;raw_response_values+=yy.size;y=yy.reshape(4,p,o);v=(y[0]-y[1]-y[2]+y[3])/(np.float64(4)*np.float64(h)*np.float64(h));computed[row,:,ii,jj]=v;computed[row,:,jj,ii]=v
  kind=['fine','coarse'][sid];saved=chunks(kind);assert computed.tobytes()==saved.tobytes(),(case,kind);hvalues[kind]=computed;del saved
 rich=(4*hvalues['fine']-hvalues['coarse'])/3;saved=chunks('richardson');assert rich.tobytes()==saved.tobytes();hvalues['richardson']=rich;del saved
 casespaces={};reference_projector=None
 for kind in ['atoms','reference','fine','coarse','richardson']:
  rows=atoms if kind=='atoms' else atomvec(hvalues[kind],n).reshape(-1,p);maximum=float(np.max(abs(rows)));normalized=rows/maximum
  U,singular,V=np.linalg.svd(normalized,full_matrices=False);rank=int(np.sum(singular>1e-10*singular[0]));values=np.pad(singular*maximum,(0,p-len(singular)));basis=V[:rank];projector=basis.T@basis
  saved=load(f'{case}-spectrum-{kind}.npz',{'singular_values':(p,),'projector':(p,p),'gate_residuals':(m,),'random_residuals':(64,)})
  close(saved['singular_values'],values,'full_singular_spectra',atol=1e-12);close(saved['projector'],projector,'projector',rtol=1e-10,atol=1e-11)
  info=stored['spaces'][kind];assert info['rank']==rank and info['full_space']==(rank==p)
  if rank==0:gap=None;reason='zero_rank'
  elif rank==p:gap=None;reason='full_space'
  elif values[rank]==0:gap=None;reason='zero_next_singular'
  else:gap=float(singular[rank-1]/singular[rank]);reason=None
  assert info['gap_reason']==reason
  if gap is None:assert info['gap'] is None
  else:close(info['gap'],gap,'gap',rtol=1e-8)
  gr=residuals(projector,g);random=residuals(projector,rd);close(saved['gate_residuals'],gr,'gate_residual',atol=1e-11);close(saved['random_residuals'],random,'random_residual',atol=1e-11)
  for key,val in [('gate_residual_min',float(saved['gate_residuals'].min())),('gate_residual_max',float(saved['gate_residuals'].max())),('random_residual_min',float(saved['random_residuals'].min())),('random_residual_max',float(saved['random_residuals'].max()))]:assert info[key]==val
  if kind=='reference':reference_projector=projector;assert rank<=min(2*m,p)
  if kind in ['fine','coarse','richardson']:
   err=stable_norm(hvalues[kind]-href)/stable_norm(href);close(info['hessian_relative_error'],err,'relative_hessian_error');close(info['reference_projector_frobenius_distance'],np.linalg.norm(projector-reference_projector),'projector_distance',atol=1e-10);assert info['hessian_error_null_reason'] is None
  casespaces[kind]={'rank':rank,'gap':info['gap'],'gap_reason':reason,'random_residual_min':info['random_residual_min'],'random_residual_max':info['random_residual_max'],'gate_residual_max':info['gate_residual_max']}
  if 'hessian_relative_error' in info:casespaces[kind]['hessian_relative_error']=info['hessian_relative_error']
  del rows,normalized,U,V,basis,projector,saved
 case_results.append({'case':case,'n':n,'p':p,'analytic_recompute_relative_error':float(np.sqrt(analytic_sq/reference_sq)),'spaces':casespaces})
 (P/'audit-progress.json').write_text(json.dumps({'completed_cases':index+1,'case':case,'elapsed_seconds':time.monotonic()-start})+'\n');del hvalues,href,rich,ctx,gg,gu,cross;gc.collect()
 assert time.monotonic()-start<170,'audit budget limit'
assert counts==1454080 and batch_count==2048 and raw_response_values==39731200
for field in ['scope_preflight','transport_preflight','worker_receipt']:
 receipt=controller[field];assert receipt['process_monitor_scope']=='process_and_children' and receipt['memory_samples']>=1 and receipt['peak_sampled_rss_bytes']<512*1024**2
assert controller['transport_preflight']['free_disk_bytes_before']>=2*1024**3 and controller['transport_preflight']['status']=='passed'
assert controller['elapsed_seconds_before_final_report']<180 and total<1024**3 and manifest.stat().st_size<1024**2
current_total=sum(p.stat().st_size for p in D.rglob('*') if p.is_file());old=dict(controller)
for key in ['worker_report','elapsed_seconds_before_final_report','artifact_bytes_before_final_report']:old.pop(key)
old.update(status='started',diagnostic_completed=False);oldsize=len((json.dumps(old,allow_nan=False,sort_keys=True)+'\n').encode());assert current_total-(D/'controller-report.json').stat().st_size+oldsize==controller['artifact_bytes_before_final_report']
result={'approved':True,'scope':'Independent saved-array diagnostic audit only; no raw forward/oracle value replay, fitting or recovery claim','controller_report_sha256':sha(D/'controller-report.json'),'worker_report_sha256':sha(W/'worker-report.json'),'artifact_manifest_sha256':sha(manifest),'marker_sha256':sha(marker),'source_commit':head,'source_files_verified':len(approval['sources']),'all16cases_and4704shards_verified':True,'counts_reserved_attempted_returned_persisted':[1454080]*4,'raw_vector_slots_recombined':counts,'batch_callback_inventory':batch_count,'raw_output_scalars':raw_response_values,'all_querypoint_rows_sign_order_diagonals_byteidentical':True,'all64probe_original_rounding_and_direction_bindings':True,'RNG_array_bindings_verified':rng_bindings,'raw_response_schemas_finite_checked':True,'fp32_outputs_exactly_stored_f64':True,'all_saved_H_and_Richardson_byteidentical_to_raw_recombination':True,'analytic_product_rule_max_absolute_error':analytic_max,'all_full_spectra_ranks_gaps_projectors_controls_metrics_verified':True,'store_bytes_including_manifest':total,'current_attempt_total_bytes':current_total,'reported_bytes_before_final':controller['artifact_bytes_before_final_report'],'worker_peak_sampled_tree_rss_bytes':controller['worker_receipt']['peak_sampled_rss_bytes'],'worker_memory_samples':controller['worker_receipt']['memory_samples'],'elapsed_before_final_report_seconds':controller['elapsed_seconds_before_final_report'],'cases':case_results,'maximum_normalized_comparison_errors':errors,'registered_contexts_invocations':0,'new_reference_forward_calls':0,'new_fits':0,'consumed941xxx_reads':0,'marker_status':'CONSUMED_NO_RETRY','private_gate':'closed','audit_elapsed_seconds':time.monotonic()-start,'audit_probe_sha256':sha(Path(__file__)),'limitations':['Saved raw oracle outputs were NOT independently re-evaluated against model forward values; their integrity, input/channel schema, commitments and downstream mathematics are verified; sourceoracle inspected only','Expected public RNG arrays were regenerated solely for bytewise binding checks; no new experimental teacher/query','All ranks/projectors/membership outcomes numerical at fixed1e-10; not exact ideal-rank or full-map identifiability claims','FDspan inflation combines finite-step truncation and runtime rounding; fullFDspace does not prove fullanalyticspace','Fullspace random rank-one residual near machine precision demonstrates failure of membership discrimination here, not global nonidentifiability','Profiles change preactivation conditioning withsqrt(n) and use distinct teachers; descriptive family effect, not causal dimension-only result','Original and audit RSS are sampled tree values, not continuous peak or parent-controller RSS; elapsed excludes final report write','Privileged weights/analytic references stay outsideGit and are not learner observations; broader bias/HVP/weaker-access/universal science stillOPEN']}
(P/'outcome-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'approved':True,'audit_sha256':sha(P/'outcome-audit.json'),'cases':16,'shards':4704,'vector_slots':counts,'raw_forward_values_requeried':False,'audit_elapsed_seconds':time.monotonic()-start}))
