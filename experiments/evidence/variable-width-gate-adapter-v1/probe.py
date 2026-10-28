"""Independent small literal adapter boundary controls. No registered data/model/RNG."""
from pathlib import Path
from dataclasses import asdict
import sys,json,hashlib,tempfile
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path(__file__).parent
sys.path[:0]=[str(R/'examples/isolated_glu'),str(R/'src')]
import variable_gate_view as m
from row_acquisition_view import RowView
from scaling_domain_store import ShardStore
from fresh_rank_common import save_json,sha

def fixture(root):
 pub=root/'public';pub.mkdir();st=ShardStore(pub/'arrays')
 for kind in ('fine','coarse','richardson'):
  for j,rows in enumerate((8,8,1)):st.save(f'c00-fp32-{kind}-h{j:02d}.npz',hessian=np.zeros((rows,1,2,2)))
 save_json(pub/'cases.json',{'schema':'modelforge-row-acquisition-observations-v1','steps':[.3,.6],'cases':[{'name':'c00-fp32','n':2,'outputs':1,'probes':17,'compute':'fp32_compute'}]})
 v=RowView(pub,sha(pub/'cases.json'),sha(pub/'arrays/manifest.jsonl'))
 for case in v.cases:v.packed(case.name)
 est=root/'estimator';est.mkdir();a=ShardStore(est/'arrays');sels=[];q=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
 for i,case in enumerate(v.cases):
  basis=a.save(f'case{i:02d}-bases.npz',**{kind+'_basis':q for kind in m.VARIANTS},**{kind+'_spectrum':np.array([5.,2.,1.]) for kind in m.VARIANTS})
  rr=[3,None,2];pp={rule+'_projector':np.zeros((3,3)) if rank is None else q[:rank].T@q[:rank] for rule,rank in zip(m.RULES,rr)}
  selected=a.save(f'case{i:02d}-choices.npz',**pp,log_gaps=np.zeros((3,2)),gap_ratios=np.ones((3,2)),eligible=np.ones((3,2)),resolution_clipped=np.zeros((3,2)),stability=np.zeros(2),resolution_floor=np.array([1e-12]),ranks=np.array([3.,-1.,2.]))
  sels.append(dict(case=case.name,index=i,meta=asdict(case),choices={rule:dict(rank=rank,certificate=False) for rule,rank in zip(m.RULES,rr)},basis=basis,selected=selected))
 seal=dict(phase='all_public_choices_complete',selections=sels,input_commitments=v.verify_unchanged(),artifact_manifest_sha256=sha(est/'arrays/manifest.jsonl'),certificate=False,new_queries=0,new_fits=0)
 def reseal():
  save_json(est/'choices-sealed.json',seal)
  save_json(est/'estimator-report.json',dict(status='completed',certificate=False,private_gate='closed',new_queries=seal['new_queries'],new_fits=0,selections=sels,input_commitments=seal['input_commitments'],artifact_manifest_sha256=seal['artifact_manifest_sha256'],seal_sha256=sha(est/'choices-sealed.json')))
  return dict(cases_sha256=sha(pub/'cases.json'),manifest_sha256=sha(pub/'arrays/manifest.jsonl'),seal_sha256=sha(est/'choices-sealed.json'),report_sha256=sha(est/'estimator-report.json'))
 return pub,est,seal,reseal,q
results=[]
for scenario in ('valid','bool_query_counter','late_archive_mutation','late_public_mutation','late_extra_array','late_extra_estroot'):
 with tempfile.TemporaryDirectory(dir=P) as td:
  pub,est,seal,again,q=fixture(Path(td));pins=again()
  if scenario=='bool_query_counter':seal['new_queries']=False;pins=again()
  original=m._digest;counter=[0]
  def digest(path,maximum=8*1024**2):
   if path==est/'choices-sealed.json':
    counter[0]+=1
    if counter[0]==2:
     if scenario=='late_archive_mutation':f=est/'arrays/case00-bases.npz';f.write_bytes(f.read_bytes()+b'x')
     if scenario=='late_public_mutation':f=pub/'arrays/c00-fp32-fine-h00.npz';f.write_bytes(f.read_bytes()+b'x')
     if scenario=='late_extra_array':(est/'arrays/extra.npz').write_bytes(b'literal')
     if scenario=='late_extra_estroot':(est/'extra.json').write_text('{}')
   return original(path,maximum)
  m._digest=digest
  try:
   try:bundle=m._load_fixture_bundle(pub,est,pins)
   except ValueError as e:
    assert scenario!='valid',(scenario,str(e));results.append(dict(case=scenario,rejected=True,error=str(e)))
   else:
    assert scenario=='valid',(scenario,'unexpected acceptance')
    job=bundle.jobs[0];assert job.job_id=='job00' and job.basis.tobytes()==q[:2].tobytes() and job.n==2 and job.rank==2
    assert not job.basis.flags.writeable;results.append(dict(case=scenario,accepted=True,exact_nontrivial_prefix=True))
  finally:m._digest=original
out={'status':'passed','source_sha256':hashlib.sha256((R/'examples/isolated_glu/variable_gate_view.py').read_bytes()).hexdigest(),'controls':results,'new_RNG_query_fit_forward':0,'registered_inputs_read':False,'scope':'component adapter literal fixtures only; not numerical selector validation'}
(P/'probe-results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
