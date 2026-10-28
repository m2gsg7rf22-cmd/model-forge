"""Literal fault injection, no real jobs/teacher/model; no optimizer run."""
from pathlib import Path
import sys,hashlib,json
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');sys.path[:0]=[str(R/'examples/isolated_glu'),str(R/'src')]
import variable_gate_search as m
from variable_gate_view import GateJob
class Sink:
 def __init__(self):self.data={}
 def write(self,name,arrays):self.data[name]={k:v.copy() for k,v in arrays.items()}
s=Sink();x=np.eye(3)[:2]
def init(job,rng,sink):
 m._emit(sink,job.job_id+'-initial-raw',raw=x)
 m._emit(sink,job.job_id+'-initial-starts',starts=x)
 return x.copy(),x.copy()
i=[0]
def refiner(x,b,steps):
 i[0]+=1
 if i[0]==2:raise FloatingPointError('literal second refinement fails')
 return x.copy()
try:m._search_fixture(GateJob('job00',np.eye(6)[[0,1,3,4]]),np.random.default_rng(811),np.random.default_rng(812),s,initializer=init,refiner=refiner)
except FloatingPointError as e:
 result={'status':'counterexample_reproduced','error':str(e),'refiner_calls':i[0],'persisted_names':list(s.data),'optimized_pre_refinement_persisted':any('optimized' in a for a in s.data.values()),'first_completed_refinement_persisted':any('refined' in a for a in s.data.values()),'source_sha256':hashlib.sha256((R/'examples/isolated_glu/variable_gate_search.py').read_bytes()).hexdigest(),'tiny_literal_rng_streams_only':True,'registered_jobs_or_arrays_used':False,'newteacher_queries_fit_or_Adam':0}
else:raise AssertionError('Injected exception missing')
assert not result['optimized_pre_refinement_persisted'] and not result['first_completed_refinement_persisted']
p=Path(__file__).with_name('partial-probe-results.json');p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
