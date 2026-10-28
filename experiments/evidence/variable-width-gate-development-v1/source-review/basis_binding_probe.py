"""Literal controller invariants with IO stubs; no registered inputs or RNG."""
from pathlib import Path
from types import SimpleNamespace
import copy,hashlib,json,sys,tempfile
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path(__file__).parent;sys.path[:0]=[str(R/'examples/isolated_glu'),str(R/'src')]
import variable_gate_controller as c
import variable_gate_common as cm
from variable_gate_inventory import upper_fields
from variable_gate_view import GateJob
from variable_gate_search import GateResult
job=GateJob('job00',np.eye(3)[:2]);out=np.array([[1.,0.]])
result=GateResult('candidate_target',out,{'base_count':1,'final_count':1,'rounds':[],'certificate':False,'coverage_certified':False,'private_gate':'closed','target_hypothesis':1})
names=c.expected_case_names('job00',result)
base={name:{k:np.zeros(sh,dtype=np.float64) for k,sh in upper_fields(name,2,2).items()} for name in names}
for name,a in base.items():
 if name.endswith('-basis'):a['basis'][:]=job.basis
 if name.endswith('rng-before') or name.endswith('rng-after'):
  for stream,state,inc in [('initial',0,1),('rescue',2,3)]:
   raw=json.dumps({'bit_generator':'PCG64','state':{'state':state,'inc':inc},'has_uint32':0,'uinteger':0},sort_keys=True,separators=(',',':')).encode()
   a[stream][:len(raw)]=np.frombuffer(raw,dtype=np.uint8);a[stream+'_length'][0]=len(raw)
 for key in ('raw','starts','optimized','refined'):
  if key in a:a[key][:]=np.tile([1.,0.],(len(a[key]),1))
 if 'refine-intent' in name:a['requested'][0]=8;a['index'][0]=8*int(name[-3:])
 if 'refine-chunk' in name:
  a['count'][0]=a['attempted'][0]=8;a['complete'][0]=1;a['index'][0]=8*int(name[-3:])
 if name.endswith('initial-merge'):
  a['order'][:]=np.arange(256);a['decision'][:]=1;a['decision'][0]=2;a['nearest_index'][:]=0;a['nearest_index'][0]=-1;a['max_cosine'][:]=1;a['max_cosine'][0]=0
 if name.endswith('-final'):a['directions']=out.copy();a['losses']=np.zeros(1);a['cosines']=np.zeros((1,1));a['count'][0]=1
 if name.endswith('view-residuals'):a['losses']=np.zeros((3,1));a['count'][0]=1
states={key:cm.decode_state(base['job00-rng-before'][key],base['job00-rng-before'][key+'_length']) for key in ('initial','rescue')}
result.report.update(job_id='job00',rng_before=states,rng_after=copy.deepcopy(states),max_retained_loss=0.)
for name,a in base.items():cm.validate_arrays(name,a,2,2)

data=copy.deepcopy(base);data['job00-basis']['basis'][:]=0
c.records=lambda *a:{name+'.npz':{} for name in names}
def read(root,name,n,rank,**kwargs):cm.validate_arrays(name,data[name],n,rank);return data[name]
c.read_trace=read
try:summary=c.validate_case(job,result,SimpleNamespace(store=SimpleNamespace(root=P/'nonexistent-stub')))
except ValueError as e:out={'incorrect_basis_rejected':True,'error':str(e)}
else:out={'incorrect_basis_rejected':False,'summary':summary}
out.update(source_sha256=hashlib.sha256((R/'examples/isolated_glu/variable_gate_controller.py').read_bytes()).hexdigest(),scope='literalIOstub;noRNG/data/query/fit')
(P/'basis-binding-probe-results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
