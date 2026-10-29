from pathlib import Path
from unittest.mock import patch
import sys,json,hashlib
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-joint-response-trace-review');sys.path.insert(0,str(R/'examples/isolated_glu'));import joint_response_trace as t
checks={};real=Path.open
class Short:
 def __init__(self,f):self.f=f
 def __enter__(self):return self
 def __exit__(self,*args):self.f.close()
 def write(self,b):return self.f.write(b[:2])
 def flush(self):self.f.flush()
 def fileno(self):return self.f.fileno()
for label,target in [('metadata','rng.json'),('manifest','manifest.jsonl')]:
 s=t.TraceStore(O/('green-'+label),1)
 def patched(p,*args,**kw):
  f=real(p,*args,**kw);return Short(f) if p==s.root/target else f
 try:
  with patch.object(Path,'open',patched):
   if label=='metadata':s.metadata('rng.json',{'literal':'abcdefgh'})
   else:s.save('initial',{f.name:np.zeros(f.shape,dtype=f.dtype) for f in s.specs['initial'].fields})
 except OSError:pass
 else:raise AssertionError('short write accepted')
 assert s.failed and not s.finished and (s.root/target).stat().st_size==2
 if label=='manifest':assert s.sizes[target]==0 and s.completed_updates==0 and s.reserved_bytes==s.specs['initial'].encoded_bound
 try:s.metadata('case.json',{})
 except ValueError:pass
 else:raise AssertionError('failed store retry accepted')
 checks['short_'+label+'_write_terminal_and_not_successfully_committed']=True
s=t.TraceStore(O/'small-success',1);values={f.name:np.zeros(f.shape,dtype=f.dtype) for f in s.specs['initial'].fields};rec=s.save('initial',values)
assert rec['bytes']==(s.root/'initial.npz').stat().st_size<=s.specs['initial'].encoded_bound and rec['sha256']==hashlib.sha256((s.root/'initial.npz').read_bytes()).hexdigest()
with np.load(s.root/'initial.npz',allow_pickle=False) as z:assert set(z.files)=={'g','u','d'}
assert len((s.root/'manifest.jsonl').read_text().splitlines())==1
checks['small_numeric_archive_schema_size_manifest_and_digest']=True
try:s.metadata('terminal.json',{'status':'completed'})
except ValueError:pass
else:raise AssertionError('partial success accepted')
assert s.failed and not s.finished
checks['incomplete_inventory_cannot_write_completed_terminal']=True
result={'passed':True,'checks':checks,'source_sha256':hashlib.sha256((R/'examples/isolated_glu/joint_response_trace.py').read_bytes()).hexdigest(),'scope':'small literal zero-array storage and IO fault injection only; no schedule/optimizer/scientific data','actual_data_reads':0,'rng_calls':0,'fits':0};(O/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'checks':len(checks)}))
