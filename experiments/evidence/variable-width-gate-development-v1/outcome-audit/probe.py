"""Independent saved-only verification. No product numerical imports or RNG objects."""
from pathlib import Path
import hashlib,io,json,platform,re,subprocess,traceback,zipfile
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
P=Path('/private/tmp/modelforge-variable-gate-outcome-audit')
O=R.parent/'experiments/variable-width-gate-development-v1-20261028-43cf7a6'
D=R.parent/'experiments/paired-row-acquisition-v1-20261028-07235ac'
SOURCE='43cf7a6cc04b88d1cf1b3071fcf1fe437dd38eae';PHASE='prepare';COMMITS={};RECORDS={};LOADED=set();MAX={}
KINDS=('fine','coarse','richardson');RULES=('strict','gap','stable')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def uniq(pairs):
 d={}
 for k,v in pairs:assert k not in d,'duplicate JSON key';d[k]=v
 return d
def canon(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def js(path):
 assert not path.is_symlink() and path.stat().st_size<=2*1024**2
 value=json.loads(path.read_text(),object_pairs_hook=uniq);canon(value);COMMITS[str(path)]=sha(path);return value
def progress(phase,**extra):
 global PHASE;PHASE=phase;(P/'progress.json').write_text(json.dumps(dict(phase=phase,**extra))+'\n')
def eq(a,b,key):assert np.array_equal(a,b),(PHASE,key)
def close(a,b,key,atol=2e-12,rtol=2e-9):
 a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(key,a.shape,b.shape)
 e=float(np.max(abs(a-b))) if a.size else 0.;MAX[key]=max(MAX.get(key,0.),e)
 assert np.allclose(a,b,atol=atol,rtol=rtol),(PHASE,key,e)
def scalar(x,low=0,high=607):
 assert x.shape==(1,) and np.isfinite(x).all() and x[0]==int(x[0]) and low<=x[0]<=high
 return int(x[0])
def manifest(root):
 path=root/'manifest.jsonl';data=path.read_bytes();assert data.endswith(b'\n') and len(data)<=1024**2
 records={}
 for line in data.splitlines():
  v=json.loads(line,object_pairs_hook=uniq);assert set(v)=={'name','bytes','sha256'} and isinstance(v['name'],str) and re.fullmatch(r'[a-zA-Z0-9_.-]+\.npz',v['name']) and v['name'] not in records
  assert type(v['bytes']) is int and 0<v['bytes']<=8*1024**2 and isinstance(v['sha256'],str) and re.fullmatch('[0-9a-f]{64}',v['sha256'])
  f=root/v['name'];assert not f.is_symlink() and f.is_file() and f.stat().st_size==v['bytes'] and sha(f)==v['sha256']
  records[v['name']]=v;COMMITS[str(f)]=v['sha256']
 assert {f.name for f in root.iterdir()}==set(records)|{'manifest.jsonl'} and len(records)<=8192
 COMMITS[str(path)]=sha(path);RECORDS[str(root)]=records;return records
def archive(root,name,fields):
 path=root/name;v=RECORDS[str(root)][name];raw=path.read_bytes();assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256']
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  names=z.namelist();assert len(names)==len(fields) and set(names)=={k+'.npy' for k in fields}
  assert sum(e.file_size for e in z.infolist())<=8*1024**2
  for key,bounds in fields.items():
   with z.open(key+'.npy') as f:
    version=np.lib.format.read_magic(f);assert version in ((1,0),(2,0));shape,_,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f))
   assert dtype==np.float64 and len(shape)==len(bounds),(name,key,shape,bounds)
   for size,limit in zip(shape,bounds):assert (size==limit if type(limit) is int else limit[0]<=size<=limit[1]),(name,key,shape,bounds)
 with np.load(io.BytesIO(raw),allow_pickle=False) as z:a={k:z[k] for k in fields}
 assert all(x.dtype==np.float64 and x.dtype.isnative and np.isfinite(x).all() for x in a.values());LOADED.add(str(path));return a
def pack(a):
 n=a.shape[-1];i,j=np.triu_indices(n);return a[...,i,j]*np.where(i==j,1.,np.sqrt(2.))
def unpack(a,n=32):
 i,j=np.triu_indices(n);m=np.zeros((n,n));m[i,j]=a/np.where(i==j,1.,np.sqrt(2.));m[j,i]=m[i,j];return m
def loss(v,b):
 q=pack(v[:,:,None]*v[:,None,:]);res=q-(q@b.T)@b;return np.sum(res*res,axis=1)
def units(a):
 assert a.ndim==2 and a.shape[1]==32
 close(np.linalg.norm(a,axis=1),np.ones(len(a)),'unit_norm',atol=1e-10,rtol=0)
def ortho(b):close(b@b.T,np.eye(len(b)),'basis_orthonormality',atol=1e-11,rtol=0)
def decode(a,k):
 n=scalar(a[k+'_length'],1,512);v=a[k];assert v.shape==(512,) and np.all(v[n:]==0) and np.all(v[:n]==np.floor(v[:n])) and np.all((v[:n]>=0)&(v[:n]<=127))
 s=json.loads(bytes(v[:n].astype(np.uint8)).decode('ascii'),object_pairs_hook=uniq)
 assert set(s)=={'bit_generator','state','has_uint32','uinteger'} and s['bit_generator']=='PCG64' and set(s['state'])=={'state','inc'}
 assert all(type(x) is int and 0<=x<2**128 for x in s['state'].values()) and s['state']['inc']%2==1
 assert type(s['has_uint32']) is int and s['has_uint32'] in (0,1) and type(s['uinteger']) is int and 0<=s['uinteger']<2**32
 return s
def est_schema(budget):
 q=min(budget*32,528);c=min(q,527)
 return ({**{k+'_basis':(q,528) for k in KINDS},**{k+'_spectrum':(528,) for k in KINDS}},
 {**{k+'_projector':(528,528) for k in RULES},'log_gaps':(3,c),'gap_ratios':(3,c),'eligible':(3,c),'resolution_clipped':(3,c),'stability':(c,),'resolution_floor':(1,),'ranks':(3,)})
def caps(name,rank):
 suffix=name.split('-',1)[1];target=rank//2
 if suffix=='basis':return {'basis':(rank,528)}
 if suffix in ('rng-before','rng-after'):return {'initial':(512,),'rescue':(512,),'initial_length':(1,),'rescue_length':(1,)}
 if suffix=='initial-raw':return {'raw':(256,32)}
 if suffix=='initial-starts':return {'starts':(256,32)}
 if suffix in ('initial-pre-refine','initial-refine'):
  d={'starts':(256,32),'optimized':(256,32),'before_losses':(256,)}
  if suffix=='initial-refine':d.update(refined=(256,32),losses=(256,))
  return d
 if suffix.endswith('-merge'):
  size=256 if suffix=='initial-merge' else 512;return {k:(size,) for k in ('order','decision','max_cosine','nearest_index','losses')}
 if suffix=='final':return {'directions':((1,607),32),'losses':((1,607),),'cosines':((1,607),(1,607)),'count':(1,)}
 if suffix=='view-residuals':return {'losses':(3,(1,607)),'count':(1,)}
 if re.fullmatch(r'(initial|r[0-3])-refine-intent[0-9]{3}',suffix) or re.fullmatch(r'r[0-3]-mix-intent[0-9]{3}',suffix):return {'index':(1,),'requested':(1,)}
 if re.fullmatch(r'(initial|r[0-3])-refine-chunk[0-9]{3}',suffix):return {'refined':(8,32),'losses':(8,),'count':(1,),'attempted':(1,),'complete':(1,),'index':(1,)}
 assert re.match(r'r[0-3]-',suffix),name
 kind=suffix[3:]
 if kind=='input':return {'known':((1,target-1),32)}
 if kind=='geometry':return {'known':((1,target-1),32),'residual':((1,rank),528),'geometry':(3,),'tangent_singular':((1,target-1),3)}
 if re.fullmatch(r'mix-pre[0-9]{3}',kind):return {'raw':(8,(1,rank-2)),'normalized':(8,(1,rank-2)),'index':(1,)}
 if re.fullmatch(r'eigen-chunk[0-9]{3}',kind):return {'values':(8,32),'vectors':(8,32,32),'indices':(8,2),'starts':(16,32),'count':(1,),'attempted':(1,),'complete':(1,),'index':(1,)}
 if kind=='mixtures':return {'raw':(256,(1,rank-2)),'normalized':(256,(1,rank-2))}
 if kind=='eigen':return {'values':(256,32),'vectors':(256,32,32),'indices':(256,2)}
 if kind in ('pre-refine','refine'):
  d={'starts':(512,32),'before_losses':(512,)}
  if kind=='refine':d.update(refined=(512,32),losses=(512,))
  return d
 raise AssertionError(('unknowntrace',name))
def merge(known,v,a,b):
 units(v);err=loss(v,b);close(a['losses'],err,'merge_recomputed_loss',atol=1e-14,rtol=1e-10)
 order=np.argsort(a['losses']);eq(a['order'],order,'merge_order');selected=list(known);decision=np.zeros(len(v));maxcos=np.zeros(len(v));nearest=np.full(len(v),-1.)
 for i in order:
  if selected:
   dots=abs(np.stack(selected)@v[i]);maxcos[i]=dots.max();nearest[i]=int(np.argmax(dots))
  if a['losses'][i]<=1e-8:
   if not selected or maxcos[i]<.9995:selected.append(v[i]);decision[i]=2
   else:decision[i]=1
 eq(a['decision'],decision,'merge_decisions');eq(a['nearest_index'],nearest,'merge_nearest');close(a['max_cosine'],maxcos,'merge_cosines',atol=2e-13,rtol=0)
 out=np.stack(selected) if selected else np.empty((0,32));return out,{'accepted':int(np.count_nonzero(decision==2)),'duplicate_rejected':int(np.count_nonzero(decision==1)),'loss_rejected':int(np.count_nonzero(decision==0))}
def validate_geometry(known,b,saved,reported):
 eq(saved['known'],known,'geometry_known');components=[];small=[]
 n=32;ii,jj=np.triu_indices(n);eye=np.eye(n);scale=np.where(ii==jj,1.,np.sqrt(2.))
 for x in known:
  linear=(x[ii,None]*eye[jj]+x[jj,None]*eye[ii])*scale[:,None]
  t=np.linalg.qr(linear,mode='reduced')[0]
  z=t-b.T@(b@t);_,sv,right=np.linalg.svd(z,full_matrices=False)
  small.append(sv[-3:][::-1]);components.extend([b@(t@right[-1]),b@(t@right[-2])])
 mat=np.stack(components);_,s,vt=np.linalg.svd(mat,full_matrices=True);r=int(np.count_nonzero(s>1e-8*s[0]));dim=len(b)-r
 assert saved['geometry'][0]==r and saved['geometry'][1]==dim
 close(saved['geometry'][2],s[-1],'component_smallest_singular',atol=2e-11,rtol=2e-6)
 close(saved['tangent_singular'],np.array(small),'tangent_smallest_singulars',atol=2e-11,rtol=2e-6)
 assert reported['component_rank']==r and reported['residual_dimension']==dim and reported['coverage_certified'] is False
 close(reported['smallest_component_singular'],s[-1],'reported_component_smallest',atol=2e-11,rtol=2e-6)
 close(reported['known_tangent_smallest_three'],small,'reported_tangent_singulars',atol=2e-11,rtol=2e-6)
 expected=vt[r:]@b;actual=saved['residual'];assert actual.shape==(max(1,dim),528)
 if dim:
  ortho(actual);close(actual-(actual@b.T)@b,np.zeros_like(actual),'residual_inside_original',atol=2e-11,rtol=0)
  close(actual@actual.T,np.eye(dim),'residual_orthonormal',atol=2e-11,rtol=0)
  close(actual.T@actual,expected.T@expected,'residual_projector',atol=2e-7,rtol=2e-6)
  close(mat@(b@actual.T),np.zeros((len(mat),dim)),'residual_orthogonal_components',atol=2e-10,rtol=0)
 return r,dim

def audit():
 progress('source_and_public_input_binding')
 revpath=R/'experiments/evidence/variable-width-gate-development-v1/source-review/final-source-review.json';review=js(revpath)
 assert sha(revpath)=='6ff724f534d1ebb4b631ebf954d429ae3d21a42ec0d4aa75979fa8566e9fbffd' and review['full_variable_gate_protocol_approved'] is True and len(review['source_sha256'])==28
 for name,h in review['source_sha256'].items():
  assert sha(R/name)==h and hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+name],cwd=R)).hexdigest()==h
 subprocess.run(['git','merge-base','--is-ancestor',SOURCE,'HEAD'],cwd=R,check=True)
 cfgpath=R/'experiments/preregistrations/variable-width-gate-development-v1.json';cfg=js(cfgpath);assert sha(cfgpath)=='f8d85c9aba42c405c3a7dfe99fec97fd577a2c91a5ebcbdd410afd1bba8a2194' and len(cfg['frozen_dependencies'])==14
 assert cfg['runtime']=={'python':platform.python_version(),'numpy':np.__version__}
 for name,h in cfg['frozen_dependencies'].items():assert review['source_sha256'][name]==h
 seeds={f'job{i:02d}':{stream:int.from_bytes(hashlib.sha256(f'modelforge-variable-gate-development-v1/job{i:02d}/{stream}'.encode()).digest(),'big') for stream in ('initial','rescue')} for i in range(6)};assert cfg['seeds']==seeds
 outer=js(O/'outer-monitor.json');controller=js(O/'controller-report.json');assert outer['status']==controller['status']=='completed' and outer['controller_sha256']==sha(O/'controller-report.json')
 assert sha(O/'controller-report.json')=='00aee710089c6225cbdc80ffcece2a23ceaa56197d4838331d147ffabdc51ad2' and sha(O/'outer-monitor.json')=='0ad71dd01fe262aba09f8422b7a798f8543bd5c17ac1ce4feb270e988bfa4511'
 assert controller['source_commit']==SOURCE and controller['config_sha256']==sha(cfgpath) and controller['review_sha256']==sha(revpath)
 for receipt in [outer['receipt'],controller['scope_receipt'],controller['transport_receipt']]:
  assert receipt['process_monitor_scope']=='process_and_children' and type(receipt['memory_samples']) is int and receipt['memory_samples']>0 and 0<receipt['peak_sampled_rss_bytes']<=512*1024**2 and receipt['memory_limit_mb']==512
 assert outer['receipt']['status']=='completed' and controller['scope_receipt']['status']=='passed' and controller['transport_receipt']['status']=='passed'
 assert controller['new_queries']==0 and controller['certificate'] is False and controller['method_promotion'] is False and controller['automatic_retry'] is False and controller['private_gate']=='closed'
 assert controller['elapsed_seconds_before_final_report']<180
 marker=O.parent/'variable-width-gate-development-v1-attempt.json';mk=js(marker)
 assert mk==dict(policy=cfg['policy'],source_commit=SOURCE,config_sha256=sha(cfgpath),review_sha256=sha(revpath),output=str(O),seeds=seeds,input_commitments=cfg['input_commitments']) and sha(marker)==controller['marker_sha256']
 smap=js(O/'seed-map.json');assert smap==dict(seeds=seeds,source_commit=SOURCE)
 adapter=js(O/'adapter-report.json');assert adapter['certificate'] is False and adapter['development_only'] is True and adapter['commitments']==cfg['input_commitments']
 pub=D/'collection/public';est=D/'estimator';cp=cfg['input_commitments']
 for name,path in [('cases_sha256',pub/'cases.json'),('manifest_sha256',pub/'arrays/manifest.jsonl'),('seal_sha256',est/'choices-sealed.json'),('report_sha256',est/'estimator-report.json')]:assert sha(path)==cp[name]
 meta=js(pub/'cases.json');seal=js(est/'choices-sealed.json');er=js(est/'estimator-report.json')
 assert meta['schema']=='modelforge-row-acquisition-observations-v1' and set(meta)=={'schema','steps','cases'} and meta['steps']==[.3,.6]
 basecases=[dict(name=f'c{i:02d}-{ch}',n=32,outputs=32,probes=17,compute='fp64' if ch=='fp64' else 'fp32_compute') for i in range(3) for ch in ('fp64','fp32')];assert meta['cases']==basecases
 cases=[dict(base,name=base['name']+f'-p{budget}',probes=budget) for base in basecases for budget in (16,17)]
 pr=manifest(pub/'arrays');es=manifest(est/'arrays');assert len(pr)==54 and len(es)==24
 assert seal['phase']=='all_public_choices_complete' and seal['selections']==er['selections'] and len(seal['selections'])==12 and sha(est/'choices-sealed.json')==er['seal_sha256'] and er['status']=='completed'
 assert seal['certificate'] is False and er['certificate'] is False and er['private_gate']=='closed'
 assert all(type(x[k]) is int and x[k]==0 for x in (seal,er) for k in ('new_queries','new_fits'))
 inp=seal['input_commitments'];assert inp==er['input_commitments']==adapter['public_inputs'] and inp==dict(cases_sha256=cp['cases_sha256'],manifest_sha256=cp['manifest_sha256'],used_input_sha256={n:v['sha256'] for n,v in pr.items()},inputs_unchanged=True)
 assert sha(est/'arrays/manifest.jsonl')==seal['artifact_manifest_sha256']==er['artifact_manifest_sha256']
 assert adapter['estimator_sha256']=={n:v['sha256'] for n,v in es.items()}
 expectedpublic={f'{b["name"]}-{k}-h{j:02d}.npz' for b in basecases for k in KINDS for j in range(3)};assert set(pr)==expectedpublic
 for name in pr:
  j=int(name[-6:-4]);a=archive(pub/'arrays',name,{'hessian':((8 if j<2 else 1),32,32,32)})['hessian'];eq(a,a.swapaxes(-1,-2),'public_H_symmetry')
 jobs=[]
 for i,case in enumerate(cases):
  q=min(case['probes']*32,528);fields,chosen=est_schema(case['probes']);a=archive(est/'arrays',f'case{i:02d}-bases.npz',fields);ch=archive(est/'arrays',f'case{i:02d}-choices.npz',chosen);sel=seal['selections'][i]
  assert sel['case']==case['name'] and type(sel['index']) is int and sel['index']==i and canon(sel['meta'])==canon(case)
  assert sel['basis']==es[f'case{i:02d}-bases.npz'] and sel['selected']==es[f'case{i:02d}-choices.npz']
  for kind in KINDS:ortho(a[kind+'_basis'])
  for j,rule in enumerate(RULES):
   k=sel['choices'][rule]['rank'];assert sel['choices'][rule]['certificate'] is False and (k is None or type(k) is int and 0<=k<=q) and ch['ranks'][j]==(-1 if k is None else k)
   proj=np.zeros((528,528)) if k is None else a['richardson_basis'][:k].T@a['richardson_basis'][:k];close(ch[rule+'_projector'],proj,'sealed_projectors',atol=1e-11,rtol=0)
  if case['probes']==17:
   rank=sel['choices']['stable']['rank'];assert type(rank) is int and 0<rank<528 and rank%2==0 and rank<=192
   job=f'job{len(jobs):02d}';assert adapter['mapping'][job]==case['name'];jobs.append(dict(id=job,rank=rank,basis=a['richardson_basis'][:rank].copy(),views=[a[k+'_basis'][:rank].copy() for k in KINDS]))
 assert len(jobs)==6 and len(adapter['mapping'])==6
 tr=manifest(O/'trace');trans=manifest(O/'transport/arrays');assert len(tr)==5064 and len(trans)==20 and sha(O/'trace/manifest.jsonl')==controller['trace_manifest_sha256']
 assert sha(O/'transport/arrays/manifest.jsonl')==controller['transport_receipt']['manifest_sha256']
 summaries=[];used=set();alljobcounts=0;allrounds=0
 for job in jobs:
  jid,r,b=job['id'],job['rank'],job['basis'];target=r//2;progress('job_saved_numeric_audit',job=jid)
  cache={}
  def get(suffix):
   name=jid+'-'+suffix
   if name not in cache:cache[name]=archive(O/'trace',name+'.npz',caps(name,r));used.add(name+'.npz')
   return cache[name]
  eq(get('basis')['basis'],b,'saved_basis_job_binding');ortho(b)
  report=js(O/'reports'/f'{jid}.json');assert report['source_commit']==SOURCE and report['job_id']==jid and report['new_queries']==0 and report['certificate'] is False and report['coverage_certified'] is False and report['private_gate']=='closed' and report['target_hypothesis']==target
  states={}
  for phase in ('before','after'):
   a=get('rng-'+phase);states[phase]={k:decode(a,k) for k in ('initial','rescue')};assert report['rng_'+phase]==states[phase]
  for k in ('initial','rescue'):assert states['before'][k]['state']['inc']==states['after'][k]['state']['inc']
  assert states['before']['initial']!=states['before']['rescue']
  raw=get('initial-raw')['raw'];starts=get('initial-starts')['starts'];close(starts,raw/np.linalg.norm(raw,axis=1,keepdims=True),'initial_raw_normalization',atol=1e-14,rtol=1e-13);units(starts)
  initial=get('initial-refine');pre=get('initial-pre-refine')
  for k in ('starts','optimized','before_losses'):eq(pre[k],initial[k],'initial_pre_aggregate')
  eq(starts,initial['starts'],'initial_starts_binding');units(initial['optimized']);units(initial['refined'])
  close(initial['before_losses'],loss(initial['optimized'],b),'initial_pre_loss',atol=1e-14,rtol=1e-10);close(initial['losses'],loss(initial['refined'],b),'initial_post_loss',atol=1e-14,rtol=1e-10)
  assert np.all(initial['losses']<=initial['before_losses']+1e-12),'refinement increasedloss'
  def check_refine(prefix,agg,num):
   for ci in range(num):
    intent=get(prefix+f'-refine-intent{ci:03d}');a=get(prefix+f'-refine-chunk{ci:03d}');assert scalar(intent['index'],0,511)==ci*8 and scalar(intent['requested'],8,8)==8
    assert scalar(a['index'],0,511)==ci*8 and scalar(a['count'],8,8)==scalar(a['attempted'],8,8)==8 and scalar(a['complete'],1,1)==1
    eq(a['refined'],agg['refined'][ci*8:(ci+1)*8],'refine_chunk_direction_aggregate');close(a['losses'],agg['losses'][ci*8:(ci+1)*8],'refine_chunk_loss_aggregate',atol=1e-14,rtol=1e-12)
   units(agg['refined'])
  check_refine('initial',initial,32);im=get('initial-merge');eq(im['losses'],initial['losses'],'initial_merge_loss_binding')
  known,initial_merge=merge(np.empty((0,32)),initial['refined'],im,b);base_count=len(known);assert type(report['base_count']) is int and report['base_count']==base_count
  rounds=[];present={int(re.match(r'job[0-5][0-5]-r([0-3])-',name)[1]) for name in tr if name.startswith(jid+'-r') and re.match(r'job[0-5][0-5]-r([0-3])-',name)}
  assert present==set(range(len(present))) and len(present)<=4
  if not 0<len(known)<target:assert not present
  geometry_failed=False
  for ri in sorted(present):
   assert 0<len(known)<target
   prefix=f'r{ri}';inp=get(prefix+'-input');eq(inp['known'],known,'round_current_input');units(known)
   rg=report['rounds'][ri];assert type(rg['round']) is int and rg['round']==ri and type(rg['input_count']) is int and rg['input_count']==len(known)
   geo=get(prefix+'-geometry');cr,dim=validate_geometry(known,b,geo,rg['geometry'])
   if cr!=2*len(known) or dim!=r-2*len(known):
    assert rg['status']=='unexpected_component_geometry' and ri==max(present);geometry_failed=True;rounds.append(dict(round=ri,input_count=len(known),status='unexpected_component_geometry',component_rank=cr,residual_dimension=dim));break
   residual=geo['residual'];mixtures=get(prefix+'-mixtures');ev=get(prefix+'-eigen');agg=get(prefix+'-refine');rpre=get(prefix+'-pre-refine')
   for k in ('starts','before_losses'):eq(rpre[k],agg[k],'rescue_pre_aggregate')
   assert mixtures['raw'].shape==mixtures['normalized'].shape==(256,dim)
   close(mixtures['normalized'],mixtures['raw']/np.linalg.norm(mixtures['raw'],axis=1,keepdims=True),'mixture_normalization',atol=1e-14,rtol=1e-13)
   for ci in range(32):
    intent=get(prefix+f'-mix-intent{ci:03d}');mx=get(prefix+f'-mix-pre{ci:03d}');ex=get(prefix+f'-eigen-chunk{ci:03d}');assert scalar(intent['index'],0,255)==scalar(mx['index'],0,255)==scalar(ex['index'],0,255)==ci*8 and scalar(intent['requested'],8,8)==8
    assert scalar(ex['count'],8,8)==scalar(ex['attempted'],8,8)==8 and scalar(ex['complete'],1,1)==1
    for k in ('raw','normalized'):eq(mx[k],mixtures[k][ci*8:(ci+1)*8],'mixture_chunk_aggregate')
    for k in ('values','vectors','indices'):eq(ex[k],ev[k][ci*8:(ci+1)*8],'eigen_chunk_aggregate')
    eq(ex['starts'],agg['starts'][ci*16:(ci+1)*16],'eigen_chunk_starts')
   for k,(w,v,ix) in enumerate(zip(ev['values'],ev['vectors'],ev['indices'])):
    mat=unpack(mixtures['normalized'][k]@residual);close(v.T@v,np.eye(32),'eigen_orthogonality',atol=2e-11,rtol=0);close(mat@v,v*w,'eigen_equations',atol=2e-11,rtol=2e-9)
    close((v*w)@v.T,mat,'eigen_reconstruction',atol=2e-11,rtol=2e-9);assert np.all(np.diff(w)>=-1e-13)
    chosen=np.argsort(abs(w))[-2:];eq(ix,chosen,'selected_extreme_eigen_indices');eq(agg['starts'][2*k:2*k+2],v[:,chosen].T,'chosen_eigen_starts')
   units(agg['starts']);check_refine(prefix,agg,64)
   close(agg['before_losses'],loss(agg['starts'],b),'rescue_pre_loss',atol=1e-14,rtol=1e-10);close(agg['losses'],loss(agg['refined'],b),'rescue_post_loss',atol=1e-14,rtol=1e-10);assert np.all(agg['losses']<=agg['before_losses']+1e-12)
   mt=get(prefix+'-merge');eq(mt['losses'],agg['losses'],'rescue_merge_loss_binding');before=len(known);known,ms=merge(known,agg['refined'],mt,b)
   assert type(rg['retained_count']) is int and rg['retained_count']==len(known)
   rounds.append(dict(round=ri,input_count=before,retained_count=len(known),component_rank=cr,residual_dimension=dim,merge=ms))
   if len(known)>=target:assert ri==max(present)
  assert len(report['rounds'])==len(rounds)
  count=len(known);status='zero_candidates' if count==0 else 'overshoot' if count>target else 'candidate_target' if count==target else 'unexpected_component_geometry' if geometry_failed else 'rounds_exhausted'
  if status=='rounds_exhausted':assert len(rounds)==4
  final=get('final');assert scalar(final['count'])==count and final['directions'].shape==(max(1,count),32) and final['losses'].shape==(max(1,count),) and final['cosines'].shape==(max(1,count),max(1,count))
  eq(final['directions'][:count],known,'final_exact_merge_trajectory')
  if count:
   units(known);err=loss(known,b);close(final['losses'],err,'final_original_basis_loss',atol=1e-14,rtol=1e-10);assert np.all(err<=1e-8)
   cos=abs(known@known.T);np.fill_diagonal(cos,0);eq(final['cosines'],cos,'final_cosines');assert np.all(cos<.9995)
   assert type(report['max_retained_loss']) is float and report['max_retained_loss']==float(final['losses'].max())
  else:
   assert all(np.all(final[k]==0) for k in ('directions','losses','cosines')) and report['max_retained_loss'] is None
  views=get('view-residuals');assert scalar(views['count'])==count and views['losses'].shape==(3,max(1,count))
  expected_views=np.stack([loss(known,bv) for bv in job['views']]) if count else np.zeros((3,1));close(views['losses'],expected_views,'public_view_residuals',atol=1e-14,rtol=1e-10)
  assert report['candidate_status']==status and type(report['final_count']) is int and report['final_count']==count
  if not rounds:assert states['before']['rescue']==states['after']['rescue']
  own={name for name in tr if name.startswith(jid+'-')};assert own=={n+'.npz' for n in cache},(jid,'unknown_or_missing_trace',own^{n+'.npz' for n in cache})
  summary=dict(job_id=jid,status='completed',candidate_status=status,rank=r,target_hypothesis=target,final_count=count,rounds=len(rounds),certificate=False,report_sha256=sha(O/'reports'/f'{jid}.json'))
  assert controller['cases'][int(jid[-2:])]==summary
  summaries.append(dict(**summary,base_count=base_count,missing_target=max(0,target-count),initial_merge=initial_merge,round_details=rounds,max_retained_loss=report['max_retained_loss'],view_loss_maxima=views['losses'].max(axis=1).tolist(),RNG_draws_or_optimizer_steps_replayed=False));alljobcounts+=status=='candidate_target';allrounds+=len(rounds)
  del cache,job,initial,pre,known,final,views
 assert used==set(tr) and len(controller['cases'])==6 and controller['candidate_target_jobs']==alljobcounts
 # Twenty preflight families match independently reconstructed maximal shapes.
 transport_fields={}
 def add(f):transport_fields[json.dumps(f,sort_keys=True)]=f
 # Basis/rng/raw/start/pre/post/merge/final/view/intent/chunk/geometry/rescuefamilies.
 add({'basis':(192,528)});add({'initial':(512,),'rescue':(512,),'initial_length':(1,),'rescue_length':(1,)})
 for name in ['job00-initial-raw','job00-initial-starts','job00-initial-pre-refine','job00-initial-refine','job00-initial-merge','job00-final','job00-view-residuals','job00-initial-refine-intent000','job00-initial-refine-chunk000','job00-r0-input','job00-r0-geometry','job00-r0-mix-pre000','job00-r0-eigen-chunk000','job00-r0-mixtures','job00-r0-eigen','job00-r0-pre-refine','job00-r0-refine','job00-r0-merge']:
  shape={k:tuple(v if type(v) is int else v[1] for v in dims) for k,dims in caps(name,192).items()};add(shape)
 assert len(transport_fields)==20
 matched=set()
 for name in trans:
  # Discover only actual declared shape, then require membership in independent20family schema set.
  raw=(O/'transport/arrays'/name).read_bytes();fields={}
  with zipfile.ZipFile(io.BytesIO(raw)) as z:
   for member in z.namelist():
    with z.open(member) as f:
     ver=np.lib.format.read_magic(f);shape,_,dt=(np.lib.format.read_array_header_1_0(f) if ver==(1,0) else np.lib.format.read_array_header_2_0(f));assert dt==np.float64;fields[member[:-4]]=shape
  key=json.dumps(fields,sort_keys=True);assert key in transport_fields and key not in matched;matched.add(key)
  vals=archive(O/'transport/arrays',name,fields);assert all(np.all(a==0) for a in vals.values())
 assert matched==set(transport_fields)
 progress('final_resource_caps_and_immutable_hashes')
 plan=cfg['inventory'];tracebytes=sum(v['bytes'] for v in tr.values())+(O/'trace/manifest.jsonl').stat().st_size
 assert len(tr)<=plan['trace_shards'] and tracebytes<=plan['trace_bytes_upper_bound'] and (O/'trace/manifest.jsonl').stat().st_size<=plan['manifest_bytes_upper_bound'] and max(v['bytes'] for v in tr.values())<=plan['largest_shard_upper_bound']
 transportbytes=sum(v['bytes'] for v in trans.values())+(O/'transport/arrays/manifest.jsonl').stat().st_size;assert transportbytes<=plan['preflight_bytes_upper_bound']
 prog=js(O/'trace-progress.json');assert prog['status']=='committed' and prog['active_job']=='job05' and prog['committed_shards']==len(tr) and prog['bytes_reserved']==tracebytes and prog['last_record']==list(tr.values())[-1]
 paths={'controller-report.json','outer-monitor.json','adapter-report.json','trace-progress.json','seed-map.json'}|{f'reports/job{i:02d}.json' for i in range(6)}|{'trace/manifest.jsonl','transport/arrays/manifest.jsonl'}|{'trace/'+n for n in tr}|{'transport/arrays/'+n for n in trans}
 assert {str(f.relative_to(O)) for f in O.rglob('*') if f.is_file()}==paths
 dirs={str(parent) for n in paths for parent in Path(n).parents if str(parent)!='.'}|{'scope-engine','transport-engine'}
 assert {str(f.relative_to(O)) for f in O.rglob('*') if f.is_dir()}<=dirs
 for f in O.rglob('*'):assert not f.is_symlink()
 metadata=[O/n for n in paths if n.endswith('.json')];assert len(metadata)<=32 and all(f.stat().st_size<=128*1024 for f in metadata)
 retained=sum((O/n).stat().st_size for n in paths);assert retained+plan['parent_retained_bytes_upper_bound']+plan['metadata_reserve_bytes']<1024**3
 assert retained<=plan['trace_bytes_upper_bound']+plan['preflight_bytes_upper_bound']+plan['metadata_reserve_bytes']
 assert sha(O/'controller-report.json')==outer['controller_sha256']
 for path,h in COMMITS.items():assert sha(Path(path))==h,(path,'changed_during_audit')
 for n,h in review['source_sha256'].items():assert sha(R/n)==h
 assert len(LOADED)==54+24+5064+20,(len(LOADED),'loadedarchives')
 result={'status':'passed','actual_source_commit':SOURCE,'signed_source_files_verified':28,'runtime_pins':14,'public_archives':54,'estimator_archives':24,'trace_archives':len(tr),'transport_archives':20,'rescue_rounds_verified':allrounds,'candidate_target_jobs':alljobcounts,'jobs':summaries,'max_numeric_differences':MAX,'controller_sha256':sha(O/'controller-report.json'),'outer_sha256':sha(O/'outer-monitor.json'),'source_config_sha256':sha(cfgpath),'review_sha256':sha(revpath),'marker_sha256':sha(marker),'source_sha256':review['source_sha256'],'artifact_commitments':COMMITS,'actual_outer_resources':outer['receipt'],'actual_inner_scope':controller['scope_receipt'],'actual_transport_resources':controller['transport_receipt'],'original_elapsed_before_final_report_seconds':controller['elapsed_seconds_before_final_report'],'retained_bytes':retained,'trace_bytes_including_manifest':tracebytes,'transport_bytes_including_manifest':transportbytes,'new_registered_RNG_objects_optimizer_eigen_solve_teacher_forward_queries_or_fit':False,'RNG_draw_or_1600_Adam_12_24refinement_path_replayed':False,'true_gate_or_coverage_evaluated':False,'certificate':False,'private_gate':'closed'}
 (P/'probe-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'passed','jobs':6,'result_sha256':sha(P/'probe-results.json')}))
try:audit()
except Exception as e:
 f={'status':'failed','phase':PHASE,'error_type':type(e).__name__,'error':str(e)[:600],'traceback':traceback.format_exc()[-1800:]};(P/'probe-failure.json').write_text(json.dumps(f,indent=2)+'\n');print(json.dumps({k:f[k] for k in ('status','phase','error_type','error')}))
