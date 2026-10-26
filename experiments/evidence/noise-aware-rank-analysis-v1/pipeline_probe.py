"""Literal synthetic files only: no actual legacy observation/truth or RNG."""
from pathlib import Path
import sys,importlib.util,json
import numpy as np
P=Path('/private/tmp/modelforge-rank-analysis-review');R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');sys.path[:0]=[str(R/'examples/isolated_glu'),str(R/'src')];sp=importlib.util.spec_from_file_location('ind_pipeline',R/'examples/isolated_glu/rank_analysis.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def fixture(root,count=1):
 root.mkdir();store=m.ShardStore(root/'source');specs=[]
 for i in range(count):
  case=['p0-t0-fp64','p0-t1-fp32'][i];meta={'n':2,'outputs':1,'probes':2};specs.append((case,meta));h=np.zeros((2,1,2,2));h[:,:,0,0]=1
  for kind in ['fine','coarse','richardson']:store.save(f'{case}-{kind}-h00.npz',hessian=h)
  reference=np.zeros_like(h);reference[0,0,0,0]=1;reference[1,0,1,1]=1;store.save(f'{case}-reference-h00.npz',hessian=reference)
  g=np.array([[1.,0.],[0.,1.],[1.,1.],[1.,-1.]]);x=np.ones((2,2));store.save(case+'-context.npz',g=g,u=g*.2,d=np.ones((1,4)),reference_g=g,reference_u=g*.2,reference_d=np.ones((1,4)),probes=x,random_directions=np.tile([[1.,1.]],(64,1)),atoms=np.zeros((8,3)),preactivations=x@g.T)
 return store,specs
source,specs=fixture(P/'component-literal-two',2);reader=m.Inputs(source.root,m.sha(source.root/'manifest.jsonl'),specs);out=P/'component-output-two';out.mkdir();store=m.ShardStore(out/'arrays');original=reader.context;reads=[]
def context(case,meta):
 seal=json.loads((out/'choices-sealed.json').read_text());assert len(seal['selections'])==2;assert len(list(store.root.glob('*-bases.npz')))==2 and len(list(store.root.glob('*-choices.npz')))==2
 assert all(any(n.startswith(c+'-'+kind) for n in reader.opened) for c,_ in specs for kind in ['fine','coarse','richardson']);reads.append(case);return original(case,meta)
reader.context=context
result=m.run_views(reader,store,out,specs);assert result['status']=='completed' and len(reads)==2 and result['new_reference_queries']==result['new_fits']==0
for evaluation in result['evaluations']:
 first=evaluation['methods'][0];assert first['rank']==1 and first['certificate'] is False and evaluation['reference_numerical_rank']==2;np.testing.assert_allclose(first['lost_signal'],np.sqrt(.5),atol=1e-14);np.testing.assert_allclose(first['extra_subspace'],0,atol=1e-14)
 controls=evaluation['methods'][3:];assert all(x['metadata']['effective_rank']==2 and x['metadata']['clipped'] and x['metadata']['same_effective_rank_as_known'] for x in controls)
 assert [x['metadata']['requested_rank'] for x in controls]==[8,4,16]
assert m.verify_manifest(store.root)['shards']==6
# Null digest cannot silently disable optional hash checking in strictNPZ reader.
source_bad,meta_bad=fixture(P/'component-null-digest');mp=source_bad.root/'manifest.jsonl';records=[json.loads(s) for s in mp.read_text().splitlines()];records[0]['sha256']=None;mp.write_text(''.join(json.dumps(r)+'\n' for r in records))
try:m.Inputs(source_bad.root,m.sha(mp),meta_bad)
except ValueError:pass
else:raise AssertionError('null digest accepted')
# After sealing, corruption of either selected input or output halts BEFORE truth.
mutation_results={}
for mode in ['observation','choice']:
 source,specs=fixture(P/('component-mutate-'+mode));reader=m.Inputs(source.root,m.sha(source.root/'manifest.jsonl'),specs);out=P/('component-mutate-output-'+mode);out.mkdir();store=m.ShardStore(out/'arrays');orig=reader.seal
 def mutate(path,report):
  orig(path,report)
  if mode=='observation':
   path=source.root/(specs[0][0]+'-fine-h00.npz')
   with path.open('wb') as f:np.savez(f,hessian=np.ones((2,1,2,2)))
  else:(store.root/report['selections'][0]['selected']['name']).write_bytes(b'changed')
 reader.seal=mutate
 try:m.run_views(reader,store,out,specs)
 except ValueError:pass
 else:raise AssertionError('mutation accepted')
 assert not any('context' in n or 'reference' in n for n in reader.opened)
 mutation_results[mode]='stopped_before_privileged_read'
report={'passed':True,'all2case_choices_saved_and_sealed_before_truth':True,'unweighted_geometric_lost_signal_sqrt_half_extra_zero_verified':True,'requested_effective_clipped_metadata_collisions_verified':True,'null_digest_schema_rejected':True,'mutations':mutation_results,'synthetic_cases':2,'actual_saved_input_reads':0,'actual_reference_or_forward_queries':0,'registered_RNG':0,'private_or941xxx_reads':0}
(P/'pipeline-probe-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
