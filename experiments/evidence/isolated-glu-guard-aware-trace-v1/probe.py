from pathlib import Path
import ast,importlib.util,json,hashlib,tempfile
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-guard-aware-trace-review')
def load(name):
 s=importlib.util.spec_from_file_location(name,R/f'examples/isolated_glu/{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('guard_aware_trace');storemod=load('guard_step_evidence');rng=np.random.default_rng(771963)
# Compare every RHS assignment for critical arithmetic variables structurally.
critical={'n','gate_norm','row_norms','eig','beta','kn','dn','bn','active','rows','sv','rank','matrix','solved','z','mu','checks','objective','passed','best','dual','length','radius','factor','clipped','pre','post','post_values','post_primal','normalized_details','original_details'}
def assignments(path,func):
 tree=ast.parse(path.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==func);out={v:[] for v in critical}
 for n in ast.walk(f):
  pairs=[]
  if isinstance(n,ast.Assign):pairs=[(t,n.value) for t in n.targets]
  elif isinstance(n,ast.AnnAssign) and n.value is not None:pairs=[(n.target,n.value)]
  for target,value in pairs:
   if isinstance(target,ast.Name) and target.id in critical:out[target.id].append(ast.dump(value))
 return out
assert assignments(R/'examples/isolated_glu/guard_aware_direction.py','cone_direction')==assignments(R/'examples/isolated_glu/guard_aware_trace.py','_compute')
cases=0;raw_checked=0
for n in (1,2,4,8):
 for clipped in (False,True):
  g=rng.normal(size=n)*(1e-3 if clipped else 1e3);Q=rng.normal(size=(n,n));K=Q.T@Q+np.eye(n);d=rng.normal(size=n);A=rng.normal(size=(min(n,4),n));before=[v.copy() for v in [g,d,K,A]];expected,info=m.frozen.cone_direction(g,d,K,A);actual,trace=m.traced_direction(g,d,K,A,proposal_id=f'p{cases}');assert np.array_equal(expected,actual)
  for k,v in info.items():
   if k!='subsets':assert trace['result'][k]==v
  beta=trace['normalization']['objective_scale'];kn=K/beta;dn=d/beta;bn=A/np.linalg.norm(A,axis=1)[:,None]
  for record in trace['subsets']:
   if record['solution'] is not None:
    z=np.array(record['solution']);mu=np.array(record['normalized_duals']);assert np.allclose(2*kn@z+dn+bn.T@mu,0,atol=1e-10);assert np.allclose(bn[record['active']]@z,0,atol=1e-10);raw_checked+=1
  for orig,now in zip(before,[g,d,K,A]):assert np.array_equal(orig,now)
  cases+=1
# Failed geometry retains the actual spectrum/rank, without stale solutions.
try:m.traced_direction(np.ones(2)*100,np.array([0.,-2.]),np.eye(2),np.array([[1.,0.],[-1.,1e-11]]))
except m.TraceError as exc:
 t=exc.trace;last=t['subsets'][-1];assert last['mask']==3 and last['status']=='aborted' and last['solution'] is None and last['normalized_duals'] is None and last['exact_original_rank']==2 and len(last['singular_values'])==2 and last['reason']=='exact_independent_near_dependency'
else:raise AssertionError('global geometry failure missing')
# Preserve exact nonfinite payloads with canonical byte metadata; never JSON NaN.
old=np.linalg.solve;payload=np.array([0x7ff8000000001234,0xfff0000000000000],dtype='<u8').view('<f8')
try:
 def nonfinite(matrix,rhs):return np.resize(payload,rhs.shape)
 np.linalg.solve=nonfinite
 try:m.traced_direction(np.ones(2),np.ones(2),np.eye(2),np.eye(2))
 except m.TraceError as exc:
  for record in exc.trace['subsets']:
   enc=record['nonfinite_solution_encoding'];assert enc['dtype']=='<f8' and enc['layout']=='z_then_mu' and enc['z_count']==enc['mu_count']==2;raw=bytes.fromhex(record['nonfinite_solution_ieee754_hex']);assert np.frombuffer(raw,dtype='<u8')[0]==0x7ff8000000001234;assert record['solution'] is None
  json.dumps(exc.trace,allow_nan=False)
 else:raise AssertionError('nonfinite solve accepted')
finally:np.linalg.solve=old
# An observer that mutates its received payload cannot mutate the internal trace.
def mutator(event):
 if 'record' in event:event['record']['status']='malicious_mutation'
_,passive=m.traced_direction(np.ones(2)*100,np.ones(2),np.eye(2),np.eye(2),event_sink=mutator);assert all(r['status']!='malicious_mutation' for r in passive['subsets'])
def broken(event):
 if event['kind']=='subset_completed':raise OSError('injected sink failure')
try:m.traced_direction(np.ones(2)*100,np.ones(2),np.eye(2),np.eye(2),event_sink=broken)
except m.TraceError as exc:assert exc.trace['failure_kind']=='trace_write' and exc.trace['linear_solves_attempted']==1 and len(exc.trace['subsets'])==1
else:raise AssertionError('sink failure continued')
with tempfile.TemporaryDirectory(dir=O,prefix='store-') as tmp:
 s=storemod.TraceStore(Path(tmp)/'valid');s.run_proposal('p',np.ones(2)*100,np.ones(2),np.eye(2),np.eye(2));receipt=s.finalize({'trial_gates':np.zeros((2,2))},{'synthetic':True});assert receipt['status']=='completed' and receipt['counter_scope']=='evidence_helper_only_not_caller_fit';manifest=json.loads((s.root/'manifest.json').read_text());entry=manifest['proposals'][0]
 for key,file in [('trace_sha256','trace.json'),('problem_sha256','problem.npz'),('journal_sha256','events.jsonl')]:assert hashlib.sha256((s.root/'proposals/p'/file).read_bytes()).hexdigest()==entry[key]
 invalid=storemod.TraceStore(Path(tmp)/'invalid')
 try:invalid.run_proposal(None,np.ones(2),np.ones(2),np.eye(2),np.eye(2))
 except ValueError:assert invalid.failed
 else:raise AssertionError('invalid id did not failclose')
 p=Path(tmp)/'partial.jsonl';p.write_bytes(b'{"ordinal":0,"proposal_id":"p","kind":"proposal_started"}\n{"ordinal":');before=p.read_bytes();inspection=storemod.inspect_journal(p,'p');assert inspection['truncated_tail'] and not inspection['successful'] and p.read_bytes()==before
result={'critical_assignment_RHS_AST_parity':True,'critical_variables_compared':sorted(critical),'synthetic_differential_cases':cases,'finite_raw_subset_solutions_equations_checked':raw_checked,'bitwise_return_and_all_nontrace_info_parity':True,'global_geometry_failure_spectrum_rank_and_null_solution_preserved':True,'canonical_nonfinite_NaN_payload_bits_preserved':True,'sink_payload_deepcopy_and_failstop_verified':True,'small_store_hashes_counters_invalid_id_and_truncated_journal_checked':True,'actual_saved_arrays_read':0,'teacher_queries':0,'actual_fits':0}
(O/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
