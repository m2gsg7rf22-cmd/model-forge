from pathlib import Path
import importlib.util,json,hashlib,tempfile,itertools
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-guard-normal-geometry-review');s=importlib.util.spec_from_file_location('geometry',R/'examples/isolated_glu/guard_normal_geometry.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rng=np.random.default_rng(337513);G=rng.normal(size=(3,3))*.3;C=rng.normal(size=(3,2,3))*.2;X=rng.normal(size=(24,3));Y=rng.normal(size=(24,2))
def forward(g,c,x):
 t=x@g.T;return np.einsum('nw,wod,nd->no',t/(1+np.exp(-t)),c,x,optimize=True)
def forbid(*args,**kwargs):raise AssertionError('direction/linear solve forbidden in diagnostic')
oldsolve=np.linalg.solve;oldcone=m.cone.cone_direction;np.linalg.solve=forbid;m.cone.cone_direction=forbid
try:
 arrays,details=m.collect(G,C,X,Y,6);assert len(details['units'])==3 and details['counts']['mapping_succeeded']==3
 expected_prediction=forward(G,C,X);error=expected_prediction-Y;assert np.allclose(arrays['prediction'],expected_prediction,rtol=1e-12,atol=1e-13)
 maxnormalerror=0.
 for i,unit in enumerate(details['units']):
  normals=[];records=unit['metrics']['records'];assert len(records)==8
  for k in range(4):
   sl=slice(k*6,(k+1)*6);x=X[sl];e=error[sl];ref=Y[sl];t=x@G[i];sig=1/(1+np.exp(-t));prime=sig+t*sig*(1-sig);branch=x@C[i].T;rel=2*np.einsum('p,po,po,pd->d',prime,e,branch,x)/np.sum(ref*ref);row,out=np.unravel_index(np.argmax(abs(e)),e.shape);ab=np.sign(e[row,out])*prime[row]*branch[row,out]*x[row];normals.extend([rel,ab]);assert records[2*k+1]['row']==k*6+row and records[2*k+1]['output']==out
  expected=np.array(normals);stored=arrays['normals'][i,:8];err=float(np.linalg.norm(expected-stored)/np.linalg.norm(expected));maxnormalerror=max(maxnormalerror,err);assert err<1e-11 and arrays['normal_counts'][i]==8 and arrays['subset_counts'][i]==256
  for record in unit['geometry']['subsets']:
   mask=record['mask'];length=int(arrays['subset_spectrum_lengths'][i,mask]);assert np.array_equal(arrays['subset_singular_values'][i,mask,:length],record['singular_values']) and arrays['subset_status_codes'][i,mask]==m.CODES[record['status']]
 # Independent spectra from row Gram eigenvalues on a separated3-row system.
 A=np.array([[1.,.2,.1],[.1,1.,.3],[.2,.1,1.]]);info=m.inspect_geometry(A);B=A/np.linalg.norm(A,axis=1)[:,None]
 for rec in info['subsets'][1:]:
  rows=B[rec['active']];expected=np.sqrt(np.maximum(np.linalg.eigvalsh(rows@rows.T),0))[::-1];assert np.allclose(expected,rec['singular_values'],rtol=1e-12,atol=1e-13);assert rec['status']=='independent_within_domain'
 near=m.inspect_geometry(np.array([[1.,0.],[-1.,1e-11]]));assert near['subsets'][3]['status']=='near_independent_outside_domain' and not near['in_domain']
 dep=m.inspect_geometry(np.array([[1.,0.],[2.,0.],[-1.,0.]]));assert dep['in_domain'] and dep['subsets'][3]['exact_original_rank']==1
 # Mapping failure preserves all units and remaining computed groups, never geometry.
 badY=Y.copy();badY[:6]=m.metrics.gradient.forward(G,C,X)[:6];badarrays,bad=m.collect(G,C,X,badY,6);assert bad['counts']['mapping_failed']==3 and bad['counts']['mapping_succeeded']==0 and np.all(badarrays['subset_counts']==0)
 for u in bad['units']:assert u['metrics']['logical_plane_count'] is None and len(u['metrics']['records'])==6 and len(u['mapping_errors'])==1 and u['mapping_errors'][0]['group']==0
 # Valid computed-zero normals preserve8 logical records and empty mask0 geometry.
 zarrays,zero=m.collect(G,np.zeros_like(C),X,Y,6);assert zero['counts']['empty_geometry']==3 and zero['counts']['geometry_in_domain']==3 and np.all(zarrays['subset_counts']==1)
 for u in zero['units']:assert len(u['metrics']['records'])==8 and u['geometry']['empty_unconstrained']
 # Persistence path, small independent fixture only; do not rerun verified worstsize engine test.
 with tempfile.TemporaryDirectory(dir=O,prefix='persist-') as tmp:
  p=Path(tmp)/'diagnostic';m.reserve(p);receipt=m.persist(p,arrays,details,{})
  assert len(json.dumps(receipt).encode())<2048;summary=json.loads((p/'evidence/summary.json').read_text());assert hashlib.sha256((p/'evidence/numeric-evidence.npz').read_bytes()).hexdigest()==receipt['numeric_sha256']==summary['evidence_sha256'];assert hashlib.sha256((p/'evidence/details.json').read_bytes()).hexdigest()==receipt['details_sha256']
finally:np.linalg.solve=oldsolve;m.cone.cone_direction=oldcone
result={'all_units_preserved':True,'independent_full_forward_and_normals_max_relative_error':maxnormalerror,'all_masks_padded_spectra_status_codes_checked':True,'separated_spectra_independent_Gram_eigen_check':True,'near_vs_exact_dependency_classification_checked':True,'failed_group_remaining_records_and_null_logical_count_preserved':True,'valid_empty_geometry_mask0_preserved':True,'small_persistence_hashes_and_receipt_checked':True,'verified_worstsize_engine_test_not_repeated':True,'no_solve_and_no_cone_direction_enforced':True,'actual_saved_arrays_read':0,'teacher_queries':0,'actual_fits':0}
(O/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
