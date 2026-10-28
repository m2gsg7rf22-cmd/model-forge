"""Independent saved-output checker. No product modules, RNG, factory or forward imports."""
import hashlib,io,json,platform,subprocess,traceback,zipfile
from pathlib import Path
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-row-acquisition-outcome-audit');O=R.parent/'experiments/paired-row-acquisition-v1-20261028-07235ac'
SOURCE='07235acd0ecaa576d57b37eefb2b4e6fc3aa0c6a';PHASE='initial';METRIC={};LOADED=set();COMMIT={};RECORDS={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def unique(pairs):
 d={}
 for k,v in pairs:
  assert k not in d,'duplicate JSON key';d[k]=v
 return d
def js(p):return json.loads(p.read_text(),object_pairs_hook=unique)
def canon(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def progress(phase,**extra):
 global PHASE;PHASE=phase;(P/'progress.json').write_text(json.dumps(dict(phase=phase,**extra))+'\n')
def close(a,b,key,atol=3e-11,rtol=3e-9):
 a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(key,a.shape,b.shape)
 diff=float(np.max(np.abs(a-b))) if a.size else 0.;METRIC[key]=max(METRIC.get(key,0.),diff)
 assert np.allclose(a,b,atol=atol,rtol=rtol),(PHASE,key,diff)
def norm(a):
 scale=float(np.abs(a).max()) if a.size else 0.;return 0. if scale==0 else float(scale*np.sqrt(np.sum((a/scale)**2)))
def rownorm(a):
 scale=np.max(np.abs(a),axis=1);return scale*np.sqrt(np.sum((a/np.where(scale==0,1.,scale)[:,None])**2,axis=1))
def manifest(directory):
 path=directory/'manifest.jsonl';b=path.read_bytes();assert b.endswith(b'\n') and len(b)<=1024**2
 d={}
 for line in b.splitlines():
  x=json.loads(line,object_pairs_hook=unique);assert set(x)=={'name','bytes','sha256'} and x['name'] not in d
  assert type(x['bytes']) is int and 0<x['bytes']<=8*1024**2 and len(x['sha256'])==64 and all(c in '0123456789abcdef' for c in x['sha256'])
  assert '/' not in x['name'] and '\\' not in x['name'];f=directory/x['name'];assert f.is_file() and not f.is_symlink() and f.stat().st_size==x['bytes'];assert sha(f)==x['sha256'];d[x['name']]=x;COMMIT[str(f)]=x['sha256']
 assert {f.name for f in directory.iterdir()}==set(d)|{'manifest.jsonl'}
 COMMIT[str(path)]=sha(path);RECORDS[str(directory)]=d;return d
def npz(directory,name,schema):
 path=directory/name;record=RECORDS[str(directory)][name];data=path.read_bytes();assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  names=z.namelist();assert len(names)==len(schema) and set(names)=={k+'.npy' for k in schema};assert sum(v.file_size for v in z.infolist())<=8*1024**2
  for k,shape in schema.items():
   with z.open(k+'.npy') as f:
    version=np.lib.format.read_magic(f);assert version in [(1,0),(2,0)];declared,_,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f));assert declared==shape and dtype==np.float64,(name,k,declared,shape)
 with np.load(io.BytesIO(data),allow_pickle=False) as z:d={k:z[k] for k in schema}
 assert all(a.dtype==np.float64 and a.dtype.isnative and np.isfinite(a).all() for a in d.values());LOADED.add(str(path));return d
def pack(h):
 n=h.shape[-1];i,j=np.triu_indices(n);return h[...,i,j]*np.where(i==j,1.,np.sqrt(2.))
def decomposition(a):
 scale=float(np.abs(a).max());_,s,v=np.linalg.svd(a/(scale if scale else 1.),full_matrices=False);full=np.pad(s,(0,a.shape[1]-len(s)));return full*scale,full/(s[0] if s[0] else 1.),v

def schemas(n,m,budget=16):
 p=n*(n+1)//2;q=min(budget*n,p);count=min(q,p-1)
 return dict(context=dict(g=(m,n),u=(m,n),d=(n,m),reference_g=(m,n),reference_u=(m,n),reference_d=(n,m),probes=(17,n),random_directions=(64,n),atoms=(2*m,p),preactivations=(17,m)),inputs={'inputs':(4*p,n)},responses={'responses':(4*p,n)},H={'hessian':(8,n,n,n)},bases={**{k+'_basis':(q,p) for k in ['fine','coarse','richardson']},**{k+'_spectrum':(p,) for k in ['fine','coarse','richardson']}},choices={**{k+'_projector':(p,p) for k in ['strict','gap','stable']},'log_gaps':(3,count),'gap_ratios':(3,count),'eligible':(3,count),'resolution_clipped':(3,count),'stability':(count,),'resolution_floor':(1,),'ranks':(3,)},evaluation=dict(reference_spectrum=(p,),reference_ranks=(3,),ranks=(6,),gate_residuals=(6,m),random_residuals=(6,64),lost_subspace=(6,),extra_subspace=(6,),reference_relative_residual=(6,),observed_relative_residual=(6,),reference_component_residuals=(6,p),qualifies=(6,),reference_sensitive=(1,)))
def Hload(directory,case,kind,n):
 return np.concatenate([npz(directory,f'{case}-{kind}-h{j:02d}.npz',{'hessian':((8 if j<2 else 1),n,n,n)})['hessian'] for j in range(3)])

def best(score,mask):
 ids=np.flatnonzero(mask);return None if len(ids)==0 else int(ids[np.argmax(score[ids])])+1

def audit():
 progress('source_and_public_only')
 reviewpath=R/'experiments/evidence/paired-row-acquisition-v1/source-review/final-source-review.json';review=js(reviewpath);assert sha(reviewpath)=='6a819f3654a696cacd9297d32a5897f0106e670c3168dc95dea4fcd84a069acf'
 assert review['full_row_acquisition_protocol_approved'] and len(review['source_sha256'])==31
 for name,h in review['source_sha256'].items():
  assert sha(R/name)==h;assert hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+name],cwd=R)).hexdigest()==h
 subprocess.run(['git','merge-base','--is-ancestor',SOURCE,'HEAD'],cwd=R,check=True)
 configpath=R/'experiments/preregistrations/paired-row-acquisition-v1.json';config=js(configpath);assert sha(configpath)=='028a7997327b6c1d307ebfed9c8dc63d996a74e8611cece61eb9b4ae769fe84d';assert len(config['frozen_dependencies'])==24
 for name,h in config['frozen_dependencies'].items():assert review['source_sha256'][name]==h
 assert config['runtime']=={'python':platform.python_version(),'numpy':np.__version__}
 outer=js(O/'outer-monitor.json');controller=js(O/'controller-report.json');assert outer['status']==controller['status']=='completed' and outer['controller_sha256']==sha(O/'controller-report.json') and controller['source_commit']==SOURCE
 assert controller['config_sha256']==sha(configpath) and controller['review_sha256']==sha(reviewpath)
 estdir=O/'estimator';public=O/'collection/public';pubdir=public/'arrays';seal=js(estdir/'choices-sealed.json');er=js(estdir/'estimator-report.json')
 assert seal['phase']=='all_public_choices_complete' and seal['selections']==er['selections'] and er['status']=='completed' and sha(estdir/'choices-sealed.json')==er['seal_sha256']
 assert seal['certificate'] is False and seal['new_queries']==seal['new_fits']==0
 inp=seal['input_commitments'];assert sha(public/'cases.json')==inp['cases_sha256'] and sha(pubdir/'manifest.jsonl')==inp['manifest_sha256']
 meta=js(public/'cases.json');assert set(meta)=={'schema','steps','cases'} and meta['steps']==[.3,.6] and meta['schema']=='modelforge-row-acquisition-observations-v1'
 basecases=meta['cases'];expected_base=[]
 for t in range(3):
  for suffix,channel in [('fp64','fp64'),('fp32','fp32_compute')]:expected_base.append(dict(name=f'c{t:02d}-{suffix}',n=32,outputs=32,probes=17,compute=channel))
 assert basecases==expected_base
 cases=[dict(base,name=base['name']+f'-p{budget}',probes=budget) for base in basecases for budget in (16,17)]
 assert len(seal['selections'])==12 and inp['inputs_unchanged']
 public_records=manifest(pubdir);est_records=manifest(estdir/'arrays');assert len(public_records)==54 and len(est_records)==24
 assert sha(estdir/'arrays/manifest.jsonl')==seal['artifact_manifest_sha256']==er['artifact_manifest_sha256'] and set(inp['used_input_sha256'])==set(public_records)
 for name,h in inp['used_input_sha256'].items():assert h==public_records[name]['sha256']
 assert {f.name for f in public.iterdir()}=={'cases.json','arrays'}
 cached=[]
 for index,case in enumerate(cases):
  name,n=case['name'],case['n'];base=basecases[index//2]['name'];budget=case['probes'];p=n*(n+1)//2;q=min(budget*n,p);count=min(q,p-1);floor=64*np.finfo(float).eps*max(budget*n,p);sch=schemas(n,1,budget)
  mats=[]
  for kind in ['fine','coarse','richardson']:
   h=Hload(pubdir,base,kind,n)[:budget];assert np.array_equal(h,h.swapaxes(-1,-2));mats.append(pack(h).reshape(budget*n,p))
  prefix_hashes=[hashlib.sha256(a[:512].tobytes()).hexdigest() for a in mats]
  if budget==17:assert prefix_hashes==cached[-1]['prefix_hashes'],(name,'paired_prefix_bytes')
  dec=[decomposition(a) for a in mats];bases=npz(estdir/'arrays',f'case{index:02d}-bases.npz',sch['bases']);chosen=npz(estdir/'arrays',f'case{index:02d}-choices.npz',sch['choices'])
  for kind,(s,ns,v) in zip(['fine','coarse','richardson'],dec):
   close(bases[kind+'_spectrum'],s,'observation_spectrum',atol=1e-14);close(bases[kind+'_basis']@bases[kind+'_basis'].T,np.eye(q),'basis_orthogonality');close(np.abs(np.sum(bases[kind+'_basis']*v,axis=1)),np.ones(q),'basis_overlap',atol=1e-9)
  ns=np.stack([v[1] for v in dec]);ratios=np.maximum(ns[:,:count],floor)/np.maximum(ns[:,1:count+1],floor);logs=np.log(ratios);eligible=ns[:,:count]>floor;clipped=ns[:,1:count+1]<floor;stability=np.zeros(count)
  for a,b in [(0,1),(0,2),(1,2)]:
   squares=(dec[a][2]@dec[b][2].T)**2;increment=np.diag(squares)+np.array([squares[k,:k].sum()+squares[:k,k].sum() for k in range(q)]);overlap=np.cumsum(increment);stability=np.maximum(stability,np.sqrt(np.maximum(0.,1-overlap[:count]/np.arange(1,count+1))))
  close(chosen['gap_ratios'],ratios,'gap_ratios',rtol=2e-12);close(chosen['log_gaps'],logs,'log_gaps');close(chosen['stability'],stability,'stability',atol=8e-8);assert np.array_equal(chosen['eligible'],eligible) and np.array_equal(chosen['resolution_clipped'],clipped) and chosen['resolution_floor'][0]==floor
  ranks=[int(np.count_nonzero(ns[2]>1e-10*ns[2,0])) if ns[2,0] else 0,best(logs[2],eligible[2]&(ratios[2]>=10)),best(logs.min(axis=0),eligible.all(axis=0)&(ratios>=10).all(axis=0)&(stability<=.01))]
  assert np.array_equal(chosen['ranks'],[-1 if x is None else x for x in ranks]);sel=seal['selections'][index];assert sel['case']==name and sel['index']==index and sel['meta']==case
  for key,kind in [('basis','bases'),('selected','choices')]:assert sel[key]==est_records[f'case{index:02d}-{kind}.npz']
  for rule,k in zip(['strict','gap','stable'],ranks):
   assert sel['choices'][rule]['rank']==k and sel['choices'][rule]['certificate'] is False
   proj=np.zeros((p,p)) if k is None else dec[2][2][:k].T@dec[2][2][:k];close(chosen[rule+'_projector'],proj,'projectors')
   if rule!='strict':assert sel['choices'][rule]['next_resolution_clipped']==(None if k is None else bool(clipped[2,k-1]) if rule=='gap' else bool(clipped[:,k-1].any()))
  cached.append(dict(case=case,prefix_hashes=prefix_hashes,ranks=ranks,right=dec[2][2],ratios=ratios,clipped=clipped,q=q,p=p))
  progress('public_choice_check',completed_cases=index+1)
 # No actual truth manifest/registration/context/reference was opened above this line.
 progress('all12_public_choices_verified_before_truth_reveal')
 collection=js(O/'collection/collection-report.json');commitpath=O/'collection/truth-commitment.json';seedpath=O/'collection/seed-commitment.json';commit=js(commitpath);seedcommit=js(seedpath)
 assert sha(commitpath)==collection['truth_commitment_sha256'] and sha(seedpath)==collection['seed_commitment_sha256'] and seedcommit['source_commit']==SOURCE and seedcommit['mapping_withheld'] is True
 registration=O/'collection/registration.json';assert sha(registration)==commit['registration_sha256']==seedcommit['sha256'];secret=js(registration);assert registration.read_bytes()==canon(secret) and set(secret)=={'seed','nonce','mapping'} and len(bytes.fromhex(secret['seed']))==len(bytes.fromhex(secret['nonce']))==32
 mapping=secret['mapping'];expected_mapping=[];seed=bytes.fromhex(secret['seed'])
 order=sorted((16,48,96),key=lambda m:hashlib.sha256(seed+f'/row-acquisition-v1/assignment/{m}'.encode()).digest())
 for m in order:expected_mapping.append(dict(id=f'c{len(expected_mapping):02d}',n=32,width=m,kind='generic'))
 assert mapping==expected_mapping
 truthdir=O/'collection/truth';evaldir=O/'evaluation/arrays';tr=manifest(truthdir);evrec=manifest(evaldir);assert len(tr)==432 and len(evrec)==12 and sha(truthdir/'manifest.jsonl')==commit['truth_manifest_sha256']
 assert commit['public_manifest_sha256']==inp['manifest_sha256'] and commit['public_cases_sha256']==inp['cases_sha256']
 evreport=js(O/'evaluation/evaluation-report.json');assert evreport['status']=='completed' and len(evreport['cases'])==12 and evreport['source_commit']==SOURCE and evreport['certificate'] is False
 table=[];vectors=batches=0;expected_truth=set();expected_public=set();pair_ctx=None
 def relative(a,b):
  scale=float(np.abs(a).max());return None if scale==0 else norm(a/scale-((a/scale)@b.T)@b)/norm(a/scale)
 def geometric(a,b):return None if len(a)==0 else norm(a-(a@b.T)@b)/np.sqrt(len(a))
 def directions(v,b):
  unit=v/np.max(np.abs(v),axis=1)[:,None];unit=unit/np.sqrt(np.sum(unit**2,axis=1))[:,None];atoms=pack(unit[:,:,None]*unit[:,None,:]);return rownorm(atoms-(atoms@b.T)@b)
 def qualification(r,sensitive,p,k,lost,extra,gm,rm):return bool(not sensitive and 0<r<p and k==r and all(v is not None and np.isfinite(v) for v in [lost,extra,gm,rm]) and 0<=lost<=.01 and 0<=extra<=.01 and 0<=gm<=.01 and rm>gm)
 for baseindex,case in enumerate(basecases):
  name,n=case['name'],case['n'];teacher=mapping[baseindex//2];m=teacher['width'];p=n*(n+1)//2;sch=schemas(n,m);ctx=npz(truthdir,name+'-context.npz',sch['context']);expected_truth.add(name+'-context.npz')
  for key in ['g','u','d']:
   expected=ctx[key].astype(np.float32).astype(np.float64) if case['compute']=='fp32_compute' else ctx[key];assert ctx['reference_'+key].tobytes()==expected.tobytes(),(name,key,'rounding')
  if baseindex%2==0:pair_ctx={k:ctx[k].copy() for k in ['g','u','d','probes','random_directions']}
  else:
   for key,a in pair_ctx.items():assert a.tobytes()==ctx[key].tobytes(),(name,key,'paired_context')
   pair_ctx=None
  if teacher['kind']=='duplicate':assert np.array_equal(ctx['g'][-1],ctx['g'][0]) and np.array_equal(ctx['u'][-1],ctx['u'][0])
  if teacher['kind']=='near_parallel':assert norm(ctx['g'][-1]-ctx['g'][0])>0 and norm(ctx['u'][-1]-ctx['u'][0])>0
  gr,ur,dr,x=ctx['reference_g'],ctx['reference_u'],ctx['reference_d'],ctx['probes'];close(ctx['preactivations'],x@gr.T,'preactivations')
  atoms=np.array([a for g,u in zip(gr,ur) for a in [pack(np.outer(g,g)),pack(np.outer(g,u)+np.outer(u,g))]]);close(ctx['atoms'],atoms,'atoms')
  href=Hload(truthdir,name,'reference',n)
  for j in range(3):expected_truth.add(f'{name}-reference-h{j:02d}.npz')
  # Independent product-rule contraction with matrix products, not model responses.
  analytic=np.empty_like(href)
  for k,point in enumerate(x):
   z=point@gr.T;up=point@ur.T;sig=1/(1+np.exp(-z));first=sig*(1+z*(1-sig));second=sig*(1-sig)*(2+z-2*z*sig)
   for out in range(n):
    gg=gr.T@(gr*(dr[out]*second*up)[:,None]);cross=gr.T@(ur*(dr[out]*first)[:,None]);hh=gg+cross+cross.T;analytic[k,out]=(hh+hh.T)/2
  close(href,analytic,'analytic_H',atol=2e-12,rtol=3e-10);METRIC['analytic_H_relative_Frobenius']=max(METRIC.get('analytic_H_relative_Frobenius',0.),norm(href-analytic)/norm(href))
  savedH={kind:Hload(pubdir,name,kind,n) for kind in ['fine','coarse','richardson']};fd=[];ii,jj=np.triu_indices(n)
  for step_index,step in enumerate([.3,.6]):
   hh=np.empty_like(href)
   for probe,point in enumerate(x):
    prefix=f'{name}-s{step_index}-x{probe:02d}';xn=prefix+'-inputs.npz';yn=prefix+'-responses.npz';xx=npz(truthdir,xn,sch['inputs'])['inputs'];yy=npz(truthdir,yn,sch['responses'])['responses'];expected_truth|={xn,yn};vectors+=len(xx);batches+=1
    if case['compute']=='fp32_compute':assert yy.tobytes()==yy.astype(np.float32).astype(np.float64).tobytes(),(name,'FP32_storage')
    expected=[]
    for signi,signj in [(1,1),(1,-1),(-1,1),(-1,-1)]:
     for i,j in zip(ii,jj):
      ei=np.zeros(n);ej=np.zeros(n);ei[i]=step;ej[j]=step;expected.append((point+ei if signi==1 else point-ei)+ej if signj==1 else (point+ei if signi==1 else point-ei)-ej)
    expected=np.array(expected);assert xx.tobytes()==expected.tobytes(),(name,step_index,probe,'stencil_rows')
    ys=yy.reshape(4,p,n);value=(ys[0]-ys[1]-ys[2]+ys[3])/(np.float64(4)*np.float64(step)*np.float64(step));hh[probe,:,ii,jj]=value;hh[probe,:,jj,ii]=value
   fd.append(hh)
  close(savedH['fine'],fd[0],'fine_FD',atol=0,rtol=0);close(savedH['coarse'],fd[1],'coarse_FD',atol=0,rtol=0);close(savedH['richardson'],(4*fd[0]-fd[1])/3,'Richardson',atol=0,rtol=0)
  for kind in savedH:
   for j in range(3):expected_public.add(f'{name}-{kind}-h{j:02d}.npz')
  for local,budget in enumerate((16,17)):
   index=2*baseindex+local;item=cached[index];condition=item['case'];q=item['q'];sch=schemas(n,m,budget)
   reference=pack(href[:budget]).reshape(budget*n,p);observed=pack(savedH['richardson'][:budget]).reshape(budget*n,p);rs,rns,rv=decomposition(reference);rr=[int(np.count_nonzero(rns>cut*rns[0])) if rns[0] else 0 for cut in [1e-11,1e-10,1e-9]];r=rr[1];sensitive=rr[0]!=rr[2];rb=rv[:r];right=item['right'];ranks=item['ranks']+[min(2*m,q,p),min(2*max(1,m//2),q,p),min(4*m,q,p)];e=npz(evaldir,f'case{index:02d}-evaluation.npz',sch['evaluation']);record=evreport['cases'][index]
   assert record['case']==condition['name'] and record['base_case']==name and record['probes']==budget and record['shared_context_sha256']==tr[name+'-context.npz']['sha256'] and record['width_for_evaluation_only']==m and record['teacher_kind']==teacher['kind'] and record['reference_ranks']==rr and record['reference_rank']==r and record['reference_sensitive']==sensitive and record['certificate'] is False
   close(e['reference_spectrum'],rs,'reference_spectrum');assert np.array_equal(e['reference_ranks'],rr) and e['reference_sensitive'][0]==sensitive and np.array_equal(e['ranks'],[-1 if k is None else k for k in ranks])
   methods=[]
   for j,(method,k) in enumerate(zip(['strict','gap','stable','known_width','wrong_width_half','wrong_width_double'],ranks)):
    mr=record['methods'][j];assert mr['method']==method and mr['rank']==k and mr['certificate'] is False and mr['reference_rank_sensitive']==sensitive
    if j<3:assert mr['metadata']=={}
    else:
     assumed=[m,max(1,m//2),2*m][j-3];assert mr['metadata']==dict(assumed_width=assumed,requested_rank=2*assumed,effective_rank=k,clipped=k!=2*assumed,same_effective_rank_as_known=k==ranks[3])
    lost=extra=gm=rm=None;reason=[]
    if k is None:
     for key in ['gate_residuals','random_residuals','lost_subspace','extra_subspace','reference_relative_residual','observed_relative_residual','reference_component_residuals']:assert np.all(e[key][j]==-1),(name,method,key)
    else:
     b=right[:k];lost=geometric(rb,b);extra=geometric(b,rb)
     metrics=dict(lost_subspace=lost,extra_subspace=extra,reference_relative_residual=relative(reference,b),observed_relative_residual=relative(observed,b))
     for key,value in metrics.items():
      if value is None:assert mr.get(key) is None and e[key][j]==-1
      else:close(mr[key],value,'report_'+key);close(e[key][j],value,'raw_'+key)
     if rs[0]>0:
      weights=rs[:len(rv)]/rs[0];weights/=np.sqrt(np.dot(weights,weights));components=np.zeros(p);components[:len(rv)]=weights*rownorm(rv-(rv@b.T)@b);close(e['reference_component_residuals'][j],components,'components');close(norm(components),metrics['reference_relative_residual'],'component_identity')
     else:assert np.all(e['reference_component_residuals'][j]==-1)
     if r>0:
      gates=directions(gr,b);random=directions(ctx['random_directions'],b);gm=float(gates.max());rm=float(random.min());close(e['gate_residuals'][j],gates,'gate_residuals');close(e['random_residuals'][j],random,'random_residuals');close(mr['gate_residual_max'],gm,'report_gate_max');close(mr['random_residual_min'],rm,'report_random_min');close(mr['random_residual_max'],float(random.max()),'report_random_max')
    passed=qualification(r,sensitive,p,k,lost,extra,gm,rm);assert mr['qualifies']==passed and e['qualifies'][j]==passed and mr['rank_error']==(None if k is None else k-r)
    if sensitive:reason.append('reference_rank_sensitive')
    if r==0:reason.append('zero_reference')
    elif r==p:reason.append('unexpected_full_reference')
    else:
     if k is None:reason.append('no_candidate')
     else:
      if k!=r:reason.append('rank_mismatch')
      if lost is None or lost>.01:reason.append('lost_subspace_exceeds_0.01')
      if extra is None or extra>.01:reason.append('extra_subspace_exceeds_0.01')
      if gm is None or gm>.01:reason.append('gate_residual_exceeds_0.01')
      if rm is None or gm is None or not rm>gm:reason.append('random_separation_failure')
    assert passed==(not reason)
    methods.append(dict(method=method,rank=k,qualifies=passed,failure_causes=reason,lost=lost,extra=extra,gate_max=gm,random_min=rm,metadata=mr['metadata']))
   structural=bool(item['ranks'][1]==q and q<p and item['clipped'][2,q-1]);table.append(dict(case=condition['name'],base_case=name,probes=budget,n=n,width=m,kind=teacher['kind'],reference_ranks=rr,reference_sensitive=sensitive,strict=item['ranks'][0],gap=item['ranks'][1],stable=item['ranks'][2],gap_selects_structural_row_rank=structural,gap_ratio_at_selected=None if item['ranks'][1] is None else float(item['ratios'][2,item['ranks'][1]-1]),gap_ratio_at_reference_rank=None if r<1 or r>item['ratios'].shape[1] else float(item['ratios'][2,r-1]),methods=methods));progress('truth_and_raw_audit',completed_cases=index+1)
   del reference,observed,e
  del ctx,savedH,fd,href,analytic
 assert set(tr)==expected_truth and set(public_records)==expected_public and vectors==430848 and batches==204
 for key in ['calls_reserved','calls_attempted','calls_returned','calls_persisted']:assert collection[key]==vectors
 assert collection['callback_batches']==batches and collection['status']=='completed' and collection['source_commit']==SOURCE
 query=js(O/'collection/query-progress.json');assert query['status']=='persisted' and all(query[k]==collection[k] for k in ['calls_reserved','calls_attempted','calls_returned','calls_persisted','callback_batches'])
 by_budget={str(b):{rule:all(next(m['qualifies'] for m in row['methods'] if m['method']==rule) for row in table if row['probes']==b) for rule in ['gap','stable']} for b in (16,17)}
 paired=[]
 for i,base in enumerate(basecases):
  a,b=table[2*i:2*i+2]
  paired.append(dict(base_case=base['name'],reference_ranks16=a['reference_ranks'],reference_ranks17=b['reference_ranks'],reference_sensitive16=a['reference_sensitive'],reference_sensitive17=b['reference_sensitive'],rules={rule:dict(pass16=next(m['qualifies'] for m in a['methods'] if m['method']==rule),pass17=next(m['qualifies'] for m in b['methods'] if m['method']==rule)) for rule in ['gap','stable']}))
 assert evreport['by_budget']==by_budget and evreport['paired']==paired
 actualrules=by_budget['17'];assert evreport['rule_passes']==controller['rule_passes']==actualrules
 progress('final_caps_and_hashes')
 transport=O/'transport/arrays';tman=manifest(transport);sch=schemas(32,96);transportschemas={
 'context.npz':sch['context'],'inputs0.npz':sch['inputs'],'responses0.npz':sch['responses'],
 'reference.npz':{'hessian':(8,32,32,32)},'reference_tail.npz':{'hessian':(1,32,32,32)},
 'bases16.npz':sch['bases'],'choices16.npz':sch['choices'],
 'bases17.npz':schemas(32,96,17)['bases'],'choices17.npz':schemas(32,96,17)['choices'],'evaluation16.npz':sch['evaluation']};assert set(tman)==set(transportschemas)
 for name,sc in transportschemas.items():npz(transport,name,sc)
 storemap={'truth':(truthdir,tr),'public':(pubdir,public_records),'estimator':(estdir/'arrays',est_records),'evaluator':(evaldir,evrec)}
 for label,(directory,records_) in storemap.items():
  limits=config['inventory_plan']['stores'][label];total=sum(v['bytes'] for v in records_.values())+(directory/'manifest.jsonl').stat().st_size;assert len(records_)==limits['shards'] and total<=limits['bytes_upper_bound'] and max(x['bytes'] for x in records_.values())<=limits['shard_limit_bytes'];assert (directory/'manifest.jsonl').stat().st_size<=limits['manifest_bytes_upper_bound']
  actual={'shards':len(records_),'bytes_reserved':total,'largest_shard_bytes':max(x['bytes'] for x in records_.values())};reported=collection['truth_inventory' if label=='truth' else 'public_inventory'] if label in ['truth','public'] else er['artifact_inventory'] if label=='estimator' else evreport['artifact_inventory'];assert actual==reported
 paths={'outer-monitor.json','controller-report.json','collection/collection-report.json','collection/registration.json','collection/query-progress.json','collection/seed-commitment.json','collection/truth-commitment.json','collection/public/cases.json','estimator/choices-sealed.json','estimator/estimator-report.json','evaluation/evaluation-report.json'}
 for directory,records_ in [(truthdir,tr),(pubdir,public_records),(estdir/'arrays',est_records),(evaldir,evrec),(transport,tman)]:
  paths.add(str((directory/'manifest.jsonl').relative_to(O)));paths.update(str((directory/n).relative_to(O)) for n in records_)
 assert {str(p.relative_to(O)) for p in O.rglob('*') if p.is_file()}==paths
 allowed_dirs={str(parent) for name in paths for parent in Path(name).parents if str(parent)!='.'}|{stage+'-engine' for stage in ['scope','transport','collection','estimation','evaluation']}
 assert {str(f.relative_to(O)) for f in O.rglob('*') if f.is_dir()}<=allowed_dirs
 for pth in O.rglob('*'):assert not pth.is_symlink()
 metadata=[O/n for n in paths if not n.endswith('.npz') and not n.endswith('manifest.jsonl')];assert len(metadata)==11 and all(f.stat().st_size<=128*1024 for f in metadata)
 retained=sum((O/n).stat().st_size for n in paths);assert retained<=config['inventory_plan']['bytes_upper_bound'] and retained+8*1024**2<1024**3
 marker=O.parent/'paired-row-acquisition-v1-attempt.json';mk=js(marker);assert mk==dict(policy=config['policy'],source_commit=SOURCE,config_sha256=sha(configpath),output=str(O)) and sha(marker)==controller['marker_sha256']
 assert outer['receipt']['status']=='completed' and outer['receipt']['process_monitor_scope']=='process_and_children' and outer['receipt']['memory_samples']>0 and outer['receipt']['peak_sampled_rss_bytes']<512*1024**2
 for stage in ['scope','transport','collection','estimation','evaluation']:
  rc=controller[stage+'_receipt'];assert rc['process_monitor_scope']=='process_and_children' and rc['memory_samples']>0 and rc['peak_sampled_rss_bytes']<512*1024**2
  if stage in ['collection','estimation','evaluation']:
   directory='estimator' if stage=='estimation' else stage;assert rc['report_sha256']==sha(O/directory/(directory+'-report.json'))
 assert controller['elapsed_seconds_before_final_report']<180 and controller['transport_receipt']['manifest_sha256']==sha(transport/'manifest.jsonl')
 for path,h in COMMIT.items():assert sha(Path(path))==h
 for name,h in review['source_sha256'].items():assert sha(R/name)==h
 scientific=sum(1 for pth in LOADED if str(transport) not in pth);assert scientific==522,(scientific,len(LOADED))
 result=dict(status='passed',actual_source_commit=SOURCE,signed_source_files_verified=31,runtime_pins=24,all12_choices_recomputed_before_truth_reveal=True,scientific_archives_schema_checked=scientific,transport_archives_schema_checked=10,vector_slots_verified=vectors,callback_batches_verified=batches,rule_passes=actualrules,by_budget=by_budget,paired=paired,public_prefix_pair_bytes_verified=True,cases=table,max_numeric_differences=METRIC,retained_bytes=retained,outer_actual_resources=outer['receipt'],controller_elapsed_before_final_report=controller['elapsed_seconds_before_final_report'],source_config_sha256=sha(configpath),controller_sha256=sha(O/'controller-report.json'),outer_sha256=sha(O/'outer-monitor.json'),marker_sha256=sha(marker),artifact_hashes=COMMIT,metadata_hashes={n:sha(O/n) for n in sorted(paths) if not n.endswith('.npz')},raw_forward_response_values_replayed=False,original_factor_probe_RNG_draws_replayed=False,new_RNG_teacher_queries_or_fit=0,certificate=False,private_gate='closed')
 (P/'probe-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(status='passed',cases=12,result_sha256=sha(P/'probe-results.json'))))
try:audit()
except Exception as exc:
 failure=dict(status='failed',phase=PHASE,error_type=type(exc).__name__,error=str(exc)[:500],traceback=traceback.format_exc()[-1000:]);(P/'probe-failure.json').write_text(json.dumps(failure,indent=2)+'\n');print(json.dumps(failure))
