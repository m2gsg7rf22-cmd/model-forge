import sys,json,tempfile,hashlib
from pathlib import Path
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
sys.path[:0]=[str(R/'src'),str(R/'examples/isolated_glu')]
import rank_analysis_controller as c
from scaling_domain_store import ShardStore
ROOT=Path(tempfile.mkdtemp(prefix='literal-controller-',dir='/private/tmp/modelforge-rank-analysis-review'))
repo=ROOT/'repository';repo.mkdir();(ROOT/'experiments').mkdir();out=ROOT/'output';w=out/'worker';w.mkdir(parents=True)
s=ShardStore(w/'arrays')
for i in range(16):
 for kind in ['bases','choices','evaluation']:s.save(f'case{i:02d}-{kind}.npz',literal=np.array([1.]))
legacy=ROOT/'literal-inputs';legacy.mkdir();cases=[x[0] for x in c.SPECS]
names={x+'-context.npz' for x in cases}|{f'{x}-{kind}-h{i:02d}.npz' for x in cases for kind in ['fine','coarse','richardson','reference'] for i in range(8)}
for name in names:(legacy/name).write_bytes(b'literal')
commit={'files':528,'input_sha256':{n:c.sha(legacy/n) for n in names}}
selections=[{'case':x,'choices':{'strict':{'rank':1,'certificate':False}}} for x in cases]
def save(p,x):p.write_text(json.dumps(x))
seal=w/'choices-sealed.json';save(seal,{'selections':selections});commit['seal_sha256']=c.sha(seal);save(w/'input-commitments.json',commit)
marker=ROOT/'experiments'/c.MARKER;save(marker,dict(policy=c.POLICY,source_commit='literal-head',config_sha256='literal-config',output=str(w)))
report=dict(status='completed',source_commit='literal-head',selections=selections,evaluations=[{'case':x} for x in cases],new_reference_queries=0,new_fits=0,certificate=False,marker=str(marker),marker_sha256=c.sha(marker),artifact_inventory=c.verify_manifest(s.root),artifact_manifest_sha256=c.sha(s.root/'manifest.jsonl'),seal_sha256=c.sha(seal),input_commitments_sha256=c.sha(w/'input-commitments.json'),selected_input_files=528)
c.check_legacy_commitments=lambda _repo:legacy
rp=w/'worker-report.json'
def validate():
 save(rp,report);return c.validate_worker(out,{'worker_report_sha256':c.sha(rp)},'literal-head','literal-config',repo)
assert validate()['status']=='completed'
checks={'full_literal_48_output_528_input_schema':True}
def reject(label,mutation,restore):
 mutation()
 try:validate()
 except ValueError:checks[label]=True
 else:raise AssertionError(label)
 restore()
old=json.loads(json.dumps(report));reject('report_seal_disagreement',lambda:report['selections'][0]['choices']['strict'].update(rank=99),lambda:report.update(json.loads(json.dumps(old))))
old=json.loads(json.dumps(report));reject('evaluation_order_mismatch',lambda:report['evaluations'].reverse(),lambda:report.update(json.loads(json.dumps(old))))
n=next(iter(names));reject('postworker_selected_input_mutation',lambda:(legacy/n).write_bytes(b'changed'),lambda:(legacy/n).write_bytes(b'literal'))
oldmarker=marker.read_bytes();reject('persistent_marker_wrong_commit',lambda:save(marker,{'wrong':True}),lambda:marker.write_bytes(oldmarker))
# Full controller flow with injected, data-free engine and source checker.
orig_verify,orig_validate=c.verify_source,c.validate_worker
cfg=ROOT/'config.json';cfg.write_text('{}');review=ROOT/'review.json';review.write_text('{}')
for mode in ['complete','worker_scope','source_after','transport']:
 calls=[];heads=iter(['literal-head','different' if mode=='source_after' else 'literal-head'])
 c.verify_source=lambda *a:({'policy':c.POLICY},next(heads))
 c.validate_worker=lambda *a:dict(status='completed',synthetic_only=True)
 def engine(*a,**k):
  calls.append(k);i=len(calls)
  assert k['memory_limit_mb']==512 and k['output_limit_bytes']==2048 and k['timeout_seconds']<=180
  assert all(k['env'][n]=='1' for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'])
  return dict(status='failed' if mode=='transport' and i==2 else 'passed',process_monitor_scope='process_only_children_unavailable' if mode=='worker_scope' and i==3 else 'process_and_children',memory_samples=1)
 dest=ROOT/mode
 try:
  result=c.run_protocol(repo,dest,cfg,review,engine=engine,clock=lambda:0.)
  assert mode=='complete' and result['analysis_completed']
 except c.ProtocolError:
  assert mode!='complete'
  result=json.loads((dest/'controller-report.json').read_text());assert result['status']=='failed' and not result['analysis_completed']
  if mode=='worker_scope':assert result['worker-engine_receipt']['process_monitor_scope']=='process_only_children_unavailable'
 checks['mocked_controller_'+mode]=True
c.verify_source,c.validate_worker=orig_verify,orig_validate
# Source binding against entirely literal non-executable dependency files and mocked Git.
for name in c.DEPENDENCIES:
 p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('literal non-executable source')
signed={n:c.sha(repo/n) for n in c.DEPENDENCIES};doc=repo/'protocol.md';doc.write_text('literal');signed['protocol.md']=c.sha(doc)
cfg=repo/'config.json';save(cfg,dict(policy=c.POLICY,frozen_dependencies={n:signed[n] for n in c.DEPENDENCIES}));signed['config.json']=c.sha(cfg)
review=ROOT/'signed-review.json';save(review,dict(rank_analysis_full_protocol_approved=True,source_sha256=signed))
oldgit=c.subprocess.check_output
c.subprocess.check_output=lambda args,**k: '' if args[1]=='status' else 'discovery-lab' if args[1]=='branch' else 'literal-head'
assert c.verify_source(repo,cfg,review)[1]=='literal-head';doc.write_text('mutated')
try:c.verify_source(repo,cfg,review)
except ValueError:checks['signed_document_mutation_rejected']=True
else:raise AssertionError('sourcegate')
c.subprocess.check_output=oldgit
result=dict(status='passed',checks=checks,registered_data_reads=0,model_forwards=0,RNG=0,private_or_941xxx_reads=0,actual_protocol_execution=False,resource_scope='tiny literal probes; no process-tree budget claim')
Path('/private/tmp/modelforge-rank-analysis-review/controller-probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
