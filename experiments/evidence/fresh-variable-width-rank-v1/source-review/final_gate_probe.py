"""Final source/seal gate probes on literal files, no subprocess or registered data."""
import sys,json,tempfile,copy,hashlib,platform
from pathlib import Path
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-fresh-protocol-review');sys.path[:0]=[str(R/'src'),str(R/'examples/isolated_glu')]
import fresh_rank_controller as c
import fresh_rank_estimation as est
import fresh_rank_evaluation as ev
from fresh_rank_view import FDView
from scaling_domain_store import ShardStore,sha
root=Path(tempfile.mkdtemp(prefix='final-literal-',dir=P));repo=root/'repository';repo.mkdir();checks={}
for name in c.REQUIRED_REVIEW:
 p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('literal non-executable signed fixture')
config=repo/'prereg.json';c.save_json(config,dict(policy=c.POLICY,inventory_plan=c.inventory_plan(),runtime={'python':platform.python_version(),'numpy':np.__version__},frozen_dependencies={n:sha(repo/n) for n in c.DEPENDENCIES}))
signed={n:sha(repo/n) for n in c.REQUIRED_REVIEW|{'prereg.json'}};review=root/'review.json';c.save_json(review,dict(full_fresh_protocol_approved=True,source_sha256=signed));original_git=c.subprocess.check_output
c.subprocess.check_output=lambda command,**kw: '' if command[1]=='status' else 'discovery-lab' if command[1]=='branch' else 'literal-head'
try:
 assert c.verify_source(repo,config,review)[1]=='literal-head';checks['complete_literal_fullsource_gate']=True
 document=repo/'docs/glu-fresh-rank-protocol.md';old=document.read_bytes();document.write_text('changed')
 try:c.verify_source(repo,config,review)
 except ValueError:checks['signed_document_mutation_rejected']=True
 else:raise AssertionError('changed document')
 document.write_bytes(old)
 r=json.loads(review.read_text());r['full_fresh_protocol_approved']=False;c.save_json(review,r)
 try:c.verify_source(repo,config,review)
 except ValueError:checks['component_only_review_rejected']=True
 else:raise AssertionError('component approval')
 r['full_fresh_protocol_approved']=True;r['source_sha256'].pop('tests/test_glu_fresh_rank_stages.py');c.save_json(review,r)
 try:c.verify_source(repo,config,review)
 except ValueError:checks['missing_required_signed_test_rejected']=True
 else:raise AssertionError('missing mandatory test')
finally:c.subprocess.check_output=original_git
public=root/'public';public.mkdir();s=ShardStore(public/'arrays');cases=[]
for suffix,channel in [('fp64','fp64'),('fp32','fp32_compute')]:
 name='c00-'+suffix
 for kind in ['fine','coarse','richardson']:s.save(name+'-'+kind+'-h00.npz',hessian=np.tile(np.eye(2),(2,1,1,1)))
 cases.append(dict(name=name,n=2,outputs=1,probes=2,compute=channel))
(public/'cases.json').write_text(json.dumps(dict(schema='modelforge-fresh-rank-observations-v1',steps=[.3,.6],cases=cases)))
truth=root/'truth';truth.mkdir();(truth/'manifest.jsonl').write_text('NEVER PARSE');(truth/'context.npz').write_bytes(b'NEVER OPEN')
view=FDView(public,sha(public/'cases.json'),sha(public/'arrays/manifest.jsonl'));out=root/'estimator';out.mkdir();store=ShardStore(out/'arrays');original_open=Path.open
reads=[]
def guarded(path,*a,**kw):
 if path==truth or truth in path.parents:reads.append(str(path));raise AssertionError('truth opened before public validation')
 return original_open(path,*a,**kw)
Path.open=guarded
try:
 report=est.estimate(view,store,out,registered=False);seal=json.loads((out/'choices-sealed.json').read_text());assert len(seal['selections'])==2 and len(seal['input_commitments']['used_input_sha256'])==6 and len(store.names)==4
 ev.validate_public_seal(out,view,seal,report);checks['literal_allcase_seal_then_public_validation_without_truth']=True
 for mode in ['last_case','last_rank_bool','last_descriptor']:
  altered=copy.deepcopy(seal)
  if mode=='last_case':altered['selections'][-1]['case']='wrong'
  elif mode=='last_rank_bool':altered['selections'][-1]['choices']['strict']['rank']=True
  else:altered['selections'][-1]['selected']['sha256']='0'*64
  er=copy.deepcopy(report);er['selections']=altered['selections']
  try:ev.validate_public_seal(out,view,altered,er)
  except ValueError:checks['before_truth_rejected_'+mode]=True
  else:raise AssertionError(mode)
 assert not reads
finally:Path.open=original_open
result={'passed':True,'checks':checks,'source_sha256':{n:sha(R/'examples/isolated_glu'/n) for n in ['fresh_rank_controller.py','fresh_rank_estimation.py','fresh_rank_evaluation.py']},'actual_subprocesses_or_factory_or_RNG_or_target_queries_or_fit_or_owneddata':0,'synthetic_cases':2,'full_protocol_executed':False}
(P/'final-gate-probe-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
