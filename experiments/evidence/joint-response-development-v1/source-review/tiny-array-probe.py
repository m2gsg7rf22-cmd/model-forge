from pathlib import Path
from unittest.mock import patch
import sys,json,io,zipfile,hashlib
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-joint-response-full-source-review');sys.path[:0]=[str(R/'examples/isolated_glu'),str(R/'src')]
import joint_response_arrays as a
from joint_response_inventory import Field,Shard
spec=Shard('literal',(Field('a','float64',(1,1)),Field('b','float64',(1,1))))
def archive(second_shape=(1,1),short=False):
 stream=io.BytesIO()
 with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as z:
  for name in ['a','b']:
   b=io.BytesIO();np.lib.format.write_array_header_1_0(b,{'descr':'<f8','fortran_order':False,'shape':second_shape if name=='b' else (1,1)})
   if not(short and name=='b'):b.write(b'\0'*8)
   z.writestr(name+'.npy',b.getvalue())
 return stream.getvalue()
value=a.decode_typed(archive(),spec);assert set(value)=={'a','b'} and all(v.shape==(1,1) and not v.flags.writeable for v in value.values())
for blob in [archive((2,2)),archive(short=True)]:
 with patch.object(np,'frombuffer',side_effect=AssertionError('array constructed before every header/payload validated')):
  try:a.decode_typed(blob,spec)
  except ValueError:pass
  else:raise AssertionError('malformed tiny header accepted')
for name in ['joint_response_runtime.py','joint_response_controller.py']:
 source=(R/'examples/isolated_glu'/name).read_text();assert 'np.load(' not in source and 'read_typed(' in source
result={'passed':True,'checks':['tiny valid two-field decode native immutable','second field wrongshape2x2 rejected before anyfrombuffer includingfirstfield','second field missing8bytepayload rejected before anyfrombuffer','both former consumers use read_typed without np.load'],'largest_literal_declared_shape':[2,2],'actual_data_reads':0,'registered_rng_calls':0,'fits':0,'large_memory_exercise_run':False,'source_sha256':{f'examples/isolated_glu/{n}.py':hashlib.sha256((R/f'examples/isolated_glu/{n}.py').read_bytes()).hexdigest() for n in ['joint_response_arrays','joint_response_runtime','joint_response_controller']}}
(O/'tiny-array-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'checks':len(result['checks'])}))
