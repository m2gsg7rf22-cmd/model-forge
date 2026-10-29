import sys,json,hashlib,dataclasses,io,zipfile,weakref,gc
from pathlib import Path
from unittest.mock import patch
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
sys.path.insert(0,str(R/'examples/isolated_glu'))
import variable_response_manifest as m
import variable_response_view as v
O=Path('/private/tmp/modelforge-variable-response-adapter-review')
T=O/'literal';T.mkdir(exist_ok=True)
raw=T/'raw';raw.mkdir(exist_ok=True)
pub=T/'public';pub.mkdir(exist_ok=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original='/literal/collection'; hashes={}
for s in range(2):
 for f in ('inputs','responses'):
  p=raw/f'c00-fp64-s{s}-x00-{f}.npz'
  np.savez(p,**{f:np.array([[1.],[2.],[3.],[4.]])+s})
  hashes[original+'/truth/'+p.name]=h(p)
(raw/'manifest.jsonl').write_text('FORBIDDEN')
(raw/'context.npz').write_text('FORBIDDEN')
counts=dict(calls_reserved=8,calls_attempted=8,calls_returned=8,calls_persisted=8,callback_batches=2)
def save(name,value):
 p=pub/(name+'.json');p.write_text(json.dumps(value));return p
c=save('collection',dict(status='completed',source_commit=m.SOURCE,**counts))
q=save('progress',dict(status='persisted',batch='c00-fp64-s1-x00',**counts))
k=save('commitments',dict(artifact_sha256=hashes,metadata_sha256={'collection/collection-report.json':h(c),'collection/query-progress.json':h(q)}))
a=save('audit',dict(approved=True,saved_outcome_audit_approved=True,actual_source_commit=m.SOURCE,private_gate='closed',artifact_commitments_sha256=h(k),vector_calls=8,callback_batches=2))
pins={n:(p,h(p)) for n,p in [('audit',a),('commitments',k),('collection',c),('progress',q)]}
manifest=m._compile(pins,m._Profile(1,1,1,('c00-fp64',)),original)
allowed={a,c,q,k}|{raw/d.name for p in manifest.pairs for d in (p.inputs,p.responses)}
opened=[]; realopen=Path.open

def guarded(p,*args,**kwargs):
 assert p in allowed,('forbidden open',str(p))
 opened.append(str(p));return realopen(p,*args,**kwargs)
def reject_list(*args,**kwargs):raise AssertionError('directory enumeration')
checks={}
with patch.object(Path,'open',guarded),patch.object(Path,'iterdir',reject_list):
 view=v._open_fixture_view(raw,manifest)
 assert len(opened)==8 and view.report()['validation']['arrays_validated']==4
 checks['all_literal_archives_before_job_exposure_and_no_adjacent_reads']=True
 g=view.jobs[0].batches;b=next(g);ref=weakref.ref(b.inputs)
 assert not b.inputs.flags.writeable
 try:b.inputs.setflags(write=True)
 except ValueError:pass
 else:raise AssertionError('mutable bytes-backed array')
 del b;gc.collect();assert ref() is None
 checks['immutable_and_idle_generator_no_array_retention']=True
 b=next(g); assert len(b.inputs)==4
 report=view.finish();assert report['status']=='completed'
 assert report['delivery']['pairs']==2 and report['completion']['arrays_validated']==0
 assert report['validation']['archives_read']==4 and report['completion']['archives_read']==4
 checks['delivery_completion_separate_counters_and_hash_only_finish']=True
# Forged descriptor must fail before any archive read.
p=manifest.pairs[0]
forged=dataclasses.replace(manifest,pairs=(dataclasses.replace(p,inputs=dataclasses.replace(p.inputs,digest='0'*64)),)+manifest.pairs[1:])
try:v._open_fixture_view(raw,forged)
except v.ViewValidationError as e:assert e.report['validation']['archives_attempted']==0
else:raise AssertionError('forged descriptor accepted')
checks['forged_manifest_rejected_before_archive_read']=True
# Last archive corruption must prevent all jobs from being exposed and preserve receipt.
last=raw/manifest.pairs[-1].responses.name;old=last.read_bytes();last.write_bytes(b'bad')
try:v._open_fixture_view(raw,manifest)
except v.ViewValidationError as e:
 assert e.report['status']=='failed' and e.report['delivery']['pairs']==0
 assert e.report['validation']['archives_attempted']==4
else:raise AssertionError('late initial corruption accepted')
last.write_bytes(old);checks['last_archive_failure_blocks_exposure_and_preserves_receipt']=True
# Mutate second pair during delivery; no partial pair counted, terminal/no retry.
view=v._open_fixture_view(raw,manifest);g=view.jobs[0].batches;next(g)
last.write_bytes(b'bad')
try:next(g)
except ValueError:pass
else:raise AssertionError('late delivery corruption accepted')
assert view.report()['status']=='failed' and view.report()['delivery']['pairs']==1
try:view.finish()
except ValueError:pass
else:raise AssertionError('failed view finish accepted')
last.write_bytes(old);checks['late_pair_failure_is_terminal_and_does_not_increment_delivery']=True
# Huge declared shape, tiny literal payload: reject schema without scientific allocation.
buf=io.BytesIO();np.lib.format.write_array_header_1_0(buf,{'descr':'<f8','fortran_order':False,'shape':(1000000000,1000000000)})
z=io.BytesIO()
with zipfile.ZipFile(z,'w',compression=zipfile.ZIP_STORED) as zz:zz.writestr('inputs.npy',buf.getvalue())
try:v._array(z.getvalue(),manifest.pairs[0].inputs)
except ValueError:pass
else:raise AssertionError('huge NPY schema accepted')
checks['huge_npy_header_rejected_before_array_construction']=True
public_allowed={R/rel for _,rel,_ in m.PUBLIC_PINS}
public_opened=[]
def public_guard(p,*args,**kwargs):
 assert p in public_allowed,('unexpected production metadata read',str(p))
 public_opened.append(str(p));return realopen(p,*args,**kwargs)
with patch.object(Path,'open',public_guard),patch.object(Path,'iterdir',reject_list):
 registered=m.load_manifest(R)
assert len(registered.pairs)==204 and len(public_opened)==4 and registered.profile.vector_rows==430848
checks['registered_public_metadata_only_compiles_204_pairs_without_transcript_or_directory_access']=True
result={'passed':True,'checks':checks,'scope':'literal n1/one case/two pairs only; no registered transcript/data/RNG/teacher/query/fit','numeric_worker':False,'source_sha256':{str(p.relative_to(R)):h(p) for p in [R/'examples/isolated_glu/variable_response_manifest.py',R/'examples/isolated_glu/variable_response_view.py']}}
(O/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'checks':len(checks),'registered_reads':0}))
