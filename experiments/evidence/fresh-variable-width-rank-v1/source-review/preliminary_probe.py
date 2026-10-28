"""Literal-only stage checks; never call contexts(), build(), entropy or any model forward."""
import sys,json,tempfile,hashlib
from pathlib import Path
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-fresh-protocol-review');sys.path[:0]=[str(R/'src'),str(R/'examples/isolated_glu')]
import fresh_rank_evaluation as e
import fresh_rank_collection as c
from scaling_domain_store import ShardStore
checks={}
ref=e.reference_summary(np.diag([1.,5e-10]));assert ref['ranks']==[2,2,1] and ref['rank']==2 and ref['sensitive'];assert e.qualify(ref,2,None,None,None,None,None) is False;checks['primary_full_sensitive_null_is_failure']=True
ref=e.reference_summary(np.eye(2));assert e.qualify(ref,2,None,None,None,None,None) is True and e.qualify(ref,2,2,0.,0.,0.,1.) is False;checks['stable_full_requires_null']=True
ref=e.reference_summary(np.zeros((2,2)));assert not e.qualify(ref,2,None,None,None,None,None);checks['zero_reference_fails_even_null']=True
ref=e.reference_summary(np.diag([1.,0.]));assert e.qualify(ref,2,1,0.01,0.01,0.01,float(np.nextafter(.01,np.inf))) and not e.qualify(ref,2,1,0.,0.,.01,.01);checks['nonfull_all_limits_and_strict_random_separation']=True
energy=getattr(e,'relative_residual',getattr(e,'energy_loss',None));assert energy is not None
assert abs(energy(np.diag([3.,4.]),np.array([[1.,0.]]))-.8)<1e-15;assert energy(np.zeros((2,2)),np.eye(2)) is None
small=energy(np.diag([1.,1e-200]),np.array([[1.,0.]]));assert small>0 and abs(small/1e-200-1)<1e-15;checks['relative_Frobenius_unsquared_point8_tiny_nonzero_preserved']=True
# Pure literal assignment only; no entropy or generation.
assignment=c.assignment(b'L'*32);assert assignment==c.assignment(b'L'*32) and len(assignment)==12
for group,expected in enumerate([(2,4,8),(4,12,20),(8,24,48),(16,48,96)]):assert sorted(v['width'] for v in assignment[3*group:3*group+3])==list(expected)
checks['deterministic_committable_literal_assignment_no_RNG']=True
# Zero context and constant callbacks are synthetic transport/counter controls, not a teacher.
root=Path(tempfile.mkdtemp(prefix='zero-counter-',dir=P));truth=ShardStore(root/'truth');public=ShardStore(root/'public');ledger=c.RawCounter(root/'progress.json',8)
ctx={'g':np.zeros((1,1)),'u':np.zeros((1,1)),'d':np.zeros((1,1)),'probes':np.zeros((1,1)),'random_directions':np.ones((64,1))};calls=[0]
def callback(x):
 calls[0]+=1
 if calls[0]==2:del ctx['random_directions']
 return np.zeros((len(x),1))
try:c.collect_case(truth,public,ledger,'literal','fp64',ctx,callback)
except ValueError:checks['deleted_context_key_detected']=True;assert ledger.terminal
else:checks['deleted_context_key_detected']=False
assert calls[0]==2 and ledger.counts['calls_attempted']==ledger.counts['calls_returned']==ledger.counts['calls_persisted']==8
result={'checks':checks,'all_checks_passed':all(checks.values()),'sources_sha256':{n:hashlib.sha256((R/'examples/isolated_glu'/n).read_bytes()).hexdigest() for n in ['fresh_rank_common.py','fresh_rank_collection.py','fresh_rank_estimation.py','fresh_rank_evaluation.py']},'synthetic_constant_callback_batches':calls[0],'synthetic_reserved_vector_slots':8,'registered_factory_or_entropy_or_RNG_invoked':False,'actual_teacher_or_target_forward_or_reference_queries':0,'actual_owned_or_private_or941xxx_reads':0,'full_protocol_approved':False,'actual_data_generation_approved':False}
(P/'preliminary-probe-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
