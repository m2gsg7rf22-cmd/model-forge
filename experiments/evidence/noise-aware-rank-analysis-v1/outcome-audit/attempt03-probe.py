try:
 """Independent saved-array audit. No estimator/worker/factory/forward/RNG imports."""
 import json,hashlib,subprocess,zipfile
 from pathlib import Path
 import numpy as np
 R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');B=R.parent/'experiments';P=Path('/private/tmp/modelforge-rank-analysis-outcome-review');O=B/'noise-aware-rank-analysis-v1-20261026-20ccebb';W=O/'worker';A=W/'arrays';L=B/'dimension-domain-v1-20261026-01b45d9'/'worker'/'arrays'
 def sha(p):
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1048576),b''):h.update(block)
  return h.hexdigest()
 def read(p):return json.loads(p.read_text())
 def close(a,b,name,atol=2e-12,rtol=3e-10):
  if not np.allclose(a,b,atol=atol,rtol=rtol):raise AssertionError(name+':'+str(np.max(np.abs(np.asarray(a)-b))))
 def npz(p,shapes=None):
  with np.load(p,allow_pickle=False) as f:
   d={k:f[k] for k in f.files}
  assert len(d)==len(f.files) and all(x.dtype==np.float64 and x.dtype.isnative and np.isfinite(x).all() for x in d.values()),str(p)
  if shapes is not None:
   assert set(d)==set(shapes),str(p)
   for k,s in shapes.items():assert d[k].shape==s,(str(p),k,d[k].shape,s)
  return d
 def pack(h):
  n=h.shape[-1];i,j=np.triu_indices(n);return h[...,i,j]*np.where(i==j,1.,np.sqrt(2.))
 def decomp(a):
  scale=np.abs(a).max();_,s,v=np.linalg.svd(a/(scale if scale else 1.),full_matrices=False);full=np.pad(s,(0,a.shape[1]-len(s)));return full*scale,full/(s[0] if s[0] else 1.),v
 cr=read(O/'controller-report.json');wr=read(W/'worker-report.json');seal=read(W/'choices-sealed.json');comm=read(W/'input-commitments.json');source=cr['source_commit']
 assert cr['status']=='completed' and cr['analysis_completed'] and wr['status']=='completed' and cr['worker_report']==wr
 assert wr['new_reference_queries']==wr['new_fits']==0 and wr['certificate'] is False
 signed=read(Path('/private/tmp/modelforge-rank-analysis-review/full-source-review.json'))['source_sha256'];assert len(signed)==17
 for name,digest in signed.items():
  assert sha(R/name)==digest,name
  assert hashlib.sha256(subprocess.check_output(['git','show',source+':'+name],cwd=R)).hexdigest()==digest,name
 subprocess.run(['git','merge-base','--is-ancestor',source,'HEAD'],cwd=R,check=True)
 old=read(Path('/private/tmp/modelforge-domain-run-review/source-review.json'));oldpins=old.get('source_sha256',old.get('sources'))
 for name,digest in oldpins.items():assert sha(R/name)==digest,name
 config=read(R/'experiments/preregistrations/noise-aware-rank-analysis-v1.json')
 for n,h in config['frozen_dependencies'].items():assert signed[n]==h
 assert cr['config_sha256']==sha(R/'experiments/preregistrations/noise-aware-rank-analysis-v1.json')
 assert wr['seal_sha256']==sha(W/'choices-sealed.json')==comm['seal_sha256'] and wr['input_commitments_sha256']==sha(W/'input-commitments.json')
 assert seal['selections']==wr['selections'] and seal['phase']=='all_observation_choices_complete' and not seal['certificate']
 cases=[f'p{p}-t{t}-{ch}' for p in range(4) for t in range(2) for ch in ['fp64','fp32']]
 assert [x['case'] for x in wr['selections']]==cases==[x['case'] for x in wr['evaluations']]
 expected={c+'-context.npz' for c in cases}|{f'{c}-{kind}-h{i:02d}.npz' for c in cases for kind in ['fine','coarse','richardson','reference'] for i in range(8)}
 assert len(expected)==comm['files']==wr['selected_input_files']==528 and set(comm['input_sha256'])==expected and comm['inputs_unchanged'] and wr['inputs_unchanged']
 manifest={x['name']:x for x in map(json.loads,(L/'manifest.jsonl').read_text().splitlines())}
 assert sha(L/'manifest.jsonl')==config['legacy']['manifest_sha256']
 for name,digest in comm['input_sha256'].items():assert sha(L/name)==digest==manifest[name]['sha256'] and (L/name).stat().st_size==manifest[name]['bytes']
 records=list(map(json.loads,(A/'manifest.jsonl').read_text().splitlines()));assert len(records)==48
 outnames={f'case{i:02d}-{kind}.npz' for i in range(16) for kind in ['bases','choices','evaluation']}
 assert {x['name'] for x in records}==outnames and {p.name for p in A.iterdir()}==outnames|{'manifest.jsonl'}
 for item in records:assert sha(A/item['name'])==item['sha256'] and (A/item['name']).stat().st_size==item['bytes']<=7*1024**2
 assert sha(A/'manifest.jsonl')==wr['artifact_manifest_sha256'] and sum(x['bytes'] for x in records)+(A/'manifest.jsonl').stat().st_size==wr['artifact_inventory']['bytes_reserved']
 marker=B/'noise-aware-rank-analysis-v1-attempt.json';mk=read(marker);assert wr['marker']==str(marker) and wr['marker_sha256']==sha(marker) and mk['source_commit']==source and mk['policy']==config['policy'] and mk['output']==str(W) and mk['config_sha256']==cr['config_sha256']
 # Original old commitments are hash-only: no old raw forward/teacher is evaluated.
 oldhash={B/'dimension-domain-v1-20261026-01b45d9'/'controller-report.json':'ec13f57abb5253522be6d9ea591392578c03a59dbeb62d711eea40a71e764dcf',B/'dimension-domain-v1-20261026-01b45d9'/'worker'/'worker-report.json':'914b731a06cba7469ba90f9d65fe5d47c2f2e698fde9f8ce68a6cdf6ac5c102f',R/'experiments/evidence/isolated-glu-dimension-domain-v1/outcome-audit.json':'14a79ddcb737ce90bede8a364e79b6967f3d1b11523f4c5b06e6f980f5fe3e35'}
 for path,digest in oldhash.items():assert sha(path)==digest
 stats={'spectrum_relative_max':0.,'basis_absolute_overlap_error_max':0.,'stability_absolute_max':0.,'projector_absolute_max':0.,'residual_absolute_max':0.,'metric_absolute_max':0.};table=[]
 for index,case in enumerate(cases):
  n=4<<(index//4);o=n;p=n*(n+1)//2;width=2*n;rows=64*o;q=min(rows,p);count=min(q,p-1);floor=64*np.finfo(float).eps*max(rows,p)
  def loadH(kind):
   chunks=[npz(L/f'{case}-{kind}-h{i:02d}.npz',{'hessian':(8,o,n,n)})['hessian'] for i in range(8)]
   h=np.concatenate(chunks);assert h.shape==(64,o,n,n);close(h,np.swapaxes(h,-1,-2),'symmetry');return pack(h).reshape(rows,p)
  views=[loadH(k) for k in ['fine','coarse','richardson']];decs=[decomp(x) for x in views];saved=npz(A/f'case{index:02d}-bases.npz',{**{k+'_basis':(q,p) for k in ['fine','coarse','richardson']},**{k+'_spectrum':(p,) for k in ['fine','coarse','richardson']}})
  for kind,(s,ns,v) in zip(['fine','coarse','richardson'],decs):
   close(saved[kind+'_spectrum'],s,'spectra',atol=1e-15);stats['spectrum_relative_max']=max(stats['spectrum_relative_max'],float(np.max(np.abs(saved[kind+'_spectrum']-s))/s[0]))
   close(saved[kind+'_basis']@saved[kind+'_basis'].T,np.eye(q),'orthonormal',atol=2e-12)
   overlap=np.abs(np.sum(saved[kind+'_basis']*v,axis=1));close(overlap,np.ones(q),'SVD vectors',atol=2e-10);stats['basis_absolute_overlap_error_max']=max(stats['basis_absolute_overlap_error_max'],float(np.max(np.abs(overlap-1))))
  c=npz(A/f'case{index:02d}-choices.npz',{**{k+'_projector':(p,p) for k in ['strict','gap','stable']},'log_gaps':(3,count),'gap_ratios':(3,count),'eligible':(3,count),'resolution_clipped':(3,count),'stability':(count,),'resolution_floor':(1,),'ranks':(3,)})
  ns=np.stack([d[1] for d in decs]);ratios=np.maximum(ns[:,:count],floor)/np.maximum(ns[:,1:count+1],floor);gaps=np.log(ratios);eligible=ns[:,:count]>floor;clipped=ns[:,1:count+1]<floor
  # Different prefix-sum construction: each k adds its new row/column overlaps.
  stability=np.zeros(count)
  for a,b in [(0,1),(0,2),(1,2)]:
   ov=(decs[a][2]@decs[b][2].T)**2;diag=np.cumsum(np.diag(ov)+np.array([ov[k,:k].sum()+ov[:k,k].sum() for k in range(q)]));distance=np.sqrt(np.maximum(0,1-diag[:count]/np.arange(1,count+1)));stability=np.maximum(stability,distance)
  close(c['gap_ratios'],ratios,'ratios');close(c['log_gaps'],gaps,'loggaps');assert np.array_equal(c['eligible'],eligible) and np.array_equal(c['resolution_clipped'],clipped);close(c['resolution_floor'],[floor],'floor',atol=0,rtol=0)
  # Near-zero distance differs under independent summation; permit only cancellation-scale absolute error.
  close(c['stability'],stability,'stability',atol=7e-8,rtol=2e-8);stats['stability_absolute_max']=max(stats['stability_absolute_max'],float(np.max(np.abs(c['stability']-stability))))
  def best(score,mask):
   possible=np.flatnonzero(mask);return None if len(possible)==0 else int(possible[np.argmax(score[possible])])+1
  ranks=[int(np.count_nonzero(ns[2]>1e-10*ns[2,0])),best(gaps[2],eligible[2]&(ratios[2]>=10)),best(gaps.min(axis=0),eligible.all(axis=0)&(ratios>=10).all(axis=0)&(stability<=.01))]
  assert np.array_equal(c['ranks'],[-1 if k is None else k for k in ranks])
  sel=wr['selections'][index];assert sel['index']==index
  for desc,kind in [('basis','bases'),('selected','choices')]:assert sel[desc]['name']==f'case{index:02d}-{kind}.npz' and sel[desc]['sha256']==sha(A/sel[desc]['name'])
  for name,k in zip(['strict','gap','stable'],ranks):
   assert sel['choices'][name]['rank']==k and sel['choices'][name]['certificate'] is False
   if name!='strict':assert sel['choices'][name]['next_resolution_clipped']==(None if k is None else bool(clipped[2,k-1]) if name=='gap' else bool(clipped[:,k-1].any()))
   projection=np.zeros((p,p)) if k is None else decs[2][2][:k].T@decs[2][2][:k];close(c[name+'_projector'],projection,'projector');stats['projector_absolute_max']=max(stats['projector_absolute_max'],float(np.max(np.abs(c[name+'_projector']-projection))))
  ref=loadH('reference');rs,rns,rv=decomp(ref);rr=int(np.count_nonzero(rns>1e-10*rns[0]));rb=rv[:rr];ev=wr['evaluations'][index];assert ev['reference_numerical_rank']==rr and ev['width_for_evaluation_only']==width
  ctx=npz(L/(case+'-context.npz'),dict(g=(width,n),u=(width,n),d=(o,width),reference_g=(width,n),reference_u=(width,n),reference_d=(o,width),probes=(64,n),random_directions=(64,n),atoms=(2*width,p),preactivations=(64,width)))
  e=npz(A/f'case{index:02d}-evaluation.npz',dict(gate_residuals=(6,width),random_residuals=(6,64),ranks=(6,),reference_rank=(1,)));assert e['reference_rank'][0]==rr
  allr=ranks+[min(2*width,q,p),min(width,q,p),min(4*width,q,p)];assert np.array_equal(e['ranks'],[-1 if k is None else k for k in allr])
  def direction(vectors,b):
   unit=vectors/np.linalg.norm(vectors,axis=1)[:,None];atoms=pack(unit[:,:,None]*unit[:,None,:]);return np.linalg.norm(atoms-(atoms@b.T)@b,axis=1)
  def rms(x,b):return None if len(x)==0 else float(np.linalg.norm(x-(x@b.T)@b)/np.sqrt(len(x)))
  for j,(method,k) in enumerate(zip(['strict','gap','stable','known_width','wrong_width_half','wrong_width_double'],allr)):
   rec=ev['methods'][j];assert rec['method']==method and rec['rank']==k and rec['certificate'] is False
   if j<3:assert rec['metadata']=={}
   else:
    assumed=[width,width//2,2*width][j-3];assert rec['metadata']==dict(assumed_width=assumed,requested_rank=2*assumed,effective_rank=k,clipped=k!=2*assumed,same_effective_rank_as_known=k==allr[3])
   if k is None:
    assert np.all(e['gate_residuals'][j]==-1) and np.all(e['random_residuals'][j]==-1) and all(rec[x] is None for x in ['rank_error','lost_signal','extra_subspace']);continue
   b=decs[2][2][:k];g=direction(ctx['reference_g'],b);random=direction(ctx['random_directions'],b)
   close(e['gate_residuals'][j],g,'gates',atol=2e-12);close(e['random_residuals'][j],random,'random',atol=2e-12);stats['residual_absolute_max']=max(stats['residual_absolute_max'],float(np.max(np.abs(e['gate_residuals'][j]-g))),float(np.max(np.abs(e['random_residuals'][j]-random))))
   metrics=dict(lost_signal=rms(rb,b),extra_subspace=rms(b,rb),gate_residual_max=float(g.max()),random_residual_min=float(random.min()),random_residual_max=float(random.max()))
   assert rec['rank_error']==k-rr and rec['full_reference_truncated']==(rr==p and k<p)
   for key,value in metrics.items():
    if value is None:assert rec[key] is None
    else:close(rec[key],value,key,atol=2e-12);stats['metric_absolute_max']=max(stats['metric_absolute_max'],abs(rec[key]-value))
  table.append(dict(case=case,reference_rank=rr,strict=ranks[0],gap=ranks[1],stable=ranks[2],gap_gate_max=ev['methods'][1].get('gate_residual_max'),gap_random_min=ev['methods'][1].get('random_residual_min'),gap_lost=ev['methods'][1].get('lost_signal')))
  (P/'audit-progress.json').write_text(json.dumps({'completed_cases':index+1,'case':case})+'\n')
  del views,decs,saved,c,ref,ctx,e
 # Rehash all selected inputs and outputs after numerical audit; immutable snapshot proof.
 for name,digest in comm['input_sha256'].items():assert sha(L/name)==digest
 for item in records:assert sha(A/item['name'])==item['sha256']
 assert sha(marker)==wr['marker_sha256'] and sha(W/'choices-sealed.json')==wr['seal_sha256']
 for key in ['scope-engine_receipt','transport-engine_receipt','worker-engine_receipt']:
  receipt=cr[key];assert receipt['process_monitor_scope']=='process_and_children' and receipt['memory_samples']>=1
 workerreceipt=cr['worker-engine_receipt'];assert workerreceipt['peak_sampled_rss_bytes']<512*1024**2 and cr['elapsed_seconds_before_final_report']<=180
 result=dict(status='passed',source_commit=source,signed_source_files=17,frozen_previous_sources=20,runtime_pins=10,input_shards=528,output_shards=48,cases=table,max_errors=stats,controller_sha256=sha(O/'controller-report.json'),worker_sha256=sha(W/'worker-report.json'),marker_sha256=sha(marker),output_manifest_sha256=sha(A/'manifest.jsonl'),input_commitments_sha256=sha(W/'input-commitments.json'),seal_sha256=sha(W/'choices-sealed.json'),actual_worker_resources=workerreceipt,actual_controller_elapsed_before_final_report=cr['elapsed_seconds_before_final_report'],new_queries=0,new_fits=0,registered_RNG=0,raw_forward_values_requeried=False,certificate=False,blind_claim=False,private_gate='closed')
 (P/'probe-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'passed','cases':16,'result_sha256':sha(P/'probe-results.json')}))
except Exception as exc:
 import traceback,json
 print(json.dumps({"status":"failed","error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()[-1400:]}))
