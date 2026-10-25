from pathlib import Path
import importlib.util,json,sys
import numpy as np
repo=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');s=importlib.util.spec_from_file_location('countdiag',repo/'examples/isolated_glu/count_diagnostic.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);original=np.arange(6,dtype=float).reshape(2,3);before=original.copy();calls=[]
def merge(known,candidates):return np.concatenate([known,candidates])
def search(known):
 out=merge(known,np.ones((1,3))*9)
 return out,[{'round':0,'input_count':len(known),'retained_count':len(out)}]
def directions():
 calls.append(True);v=np.eye(3);residual=np.arange(3,dtype=float);initial_loss=residual+1;b=np.eye(3);known=merge(original,np.ones((1,3))*8);known,_=search(known)
 assert len(v)==len(residual)==len(initial_loss)==len(b)==3
 raise ValueError('Synthetic exactcount rejection')
previous=sys.getprofile();seen=[]
def old_profile(frame,event,arg):seen.append(event)
try:
 sys.setprofile(old_profile);value,error,events,snapshot=m.capture(directions,{directions.__code__:'directions',merge.__code__:'merge',search.__code__:'search'});assert sys.getprofile() is old_profile
finally:sys.setprofile(previous)
assert len(calls)==1 and value is None and error=='Synthetic exactcount rejection';assert np.array_equal(original,before);assert set(snapshot)=={'known','v','residual','initial_loss','b'};assert [e['role'] for e in events]==['merge','merge','search','directions'];assert events[-1]['final_count']==4;snapshot['known'][0]=100;assert np.array_equal(original,before)
def unexpected():raise RuntimeError('Unexpected workerfailure')
try:m.capture(unexpected,{})
except RuntimeError:pass
else:raise AssertionError('Unexpected exception swallowed')
assert sys.getprofile() is previous
basis=np.eye(5)[:2];angle=.43;other=basis.copy();other[1]=[0,np.cos(angle),np.sin(angle),0,0];distance=m.distance(basis,other);assert np.isclose(distance,np.sin(angle)/np.sqrt(2),atol=1e-15)
result={'fake_operation_calls':len(calls),'exact_role_return_events':[e['role'] for e in events],'final_count_on_exception':events[-1]['final_count'],'copies_do_not_mutate_operation_inputs':True,'five_snapshot_arrays_captured':True,'preexisting_profiler_restored':True,'unexpected_exception_propagates_and_profiler_restored':True,'known_angle_rms_sine':distance,'expected_rms_sine':float(np.sin(angle)/np.sqrt(2)),'no_real_kernel_replay_fitting_queries_or_private_access':True};Path('/private/tmp/modelforge-count-diagnostic-review/probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
