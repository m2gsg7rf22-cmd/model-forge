"""Small literal supplied-array and mocked-controller probes, no registered RNG."""
import sys,importlib.util,json
from pathlib import Path
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-domain-run-review');sys.path.insert(0,str(R/'examples/isolated_glu'));sp=importlib.util.spec_from_file_location('ind_run_final',R/'examples/isolated_glu/scaling_domain_run.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
c=dict(g=np.array([[.2,.3],[.1,-.2]]),u=np.array([[.3,-.1],[.4,.2]]),d=np.array([[.1,.2],[-.2,.3]]),probes=np.array([[.1,.2],[-.3,.4]]),random_directions=np.array([[1.,0.],[1.,1.]]))
def query(x,channel):
 dt=np.float32 if channel=='fp32_compute' else np.float64;xx=x.astype(dt);g,u,d=(c[k].astype(dt) for k in ['g','u','d']);z=xx@g.T;return ((z/(1+np.exp(-z))*(xx@u.T))@d.T).astype(np.float64)
def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
cases={}
for channel in ['fp64','fp32_compute']:
 root=P/('case-'+channel);root.mkdir();store=m.ShardStore(root/'arrays');ledger=m.QueryLedger(root/'progress.json',48);observed=[]
 def callback(x):
  progress=json.loads(ledger.path.read_text());assert progress['calls_attempted']==12*(len(observed)+1);observed.append(x.copy());return query(x,channel)
 report=m.run_case(store,ledger,'literal',query=callback,channel=channel,**c);assert ledger.counts=={'calls_reserved':48,'calls_attempted':48,'calls_returned':48,'calls_persisted':48}
 assert m.verify_manifest(store.root)['shards']==18
 context=load(store.root/'literal-context.npz');dt=np.float32 if channel=='fp32_compute' else np.float64
 for key in ['g','u','d']:assert context['reference_'+key].tobytes()==c[key].astype(dt).astype(np.float64).tobytes()
 hs=[]
 for stepid,h in enumerate([.3,.6]):
  hh=[]
  for i,x in enumerate(c['probes']):
   xx=load(store.root/f'literal-s{stepid}-x{i:02d}-inputs.npz')['inputs'];y=load(store.root/f'literal-s{stepid}-x{i:02d}-responses.npz')['responses'];assert xx.shape==(12,2);assert xx.tobytes()==observed[stepid*2+i].tobytes();np.testing.assert_array_equal(y,query(xx,channel));yy=y.reshape(4,3,2);v=(yy[0]-yy[1]-yy[2]+yy[3])/(4*h*h);H=np.empty((2,2,2));H[:,0,0]=v[0];H[:,0,1]=H[:,1,0]=v[1];H[:,1,1]=v[2];hh.append(H)
  hh=np.array(hh);np.testing.assert_array_equal(hh,load(store.root/f'literal-{["fine","coarse"][stepid]}-h00.npz')['hessian']);hs.append(hh)
 np.testing.assert_array_equal((4*hs[0]-hs[1])/3,load(store.root/'literal-richardson-h00.npz')['hessian'])
 for kind in ['atoms','reference','fine','coarse','richardson']:
  z=load(store.root/f'literal-spectrum-{kind}.npz');assert len(z['singular_values'])==3;proj=z['projector'];np.testing.assert_allclose(proj,proj.T,atol=1e-12);np.testing.assert_allclose(proj@proj,proj,atol=1e-12)
  if kind=='atoms':rows=context['atoms']
  else:
   H=load(store.root/f'literal-{kind}-h00.npz')['hessian'];rows=np.stack([H[...,0,0],np.sqrt(2)*H[...,0,1],H[...,1,1]],axis=-1).reshape(-1,3)
  _,sv,vh=np.linalg.svd(rows/np.max(np.abs(rows)),full_matrices=False);rank=int(np.sum(sv>sv[0]*1e-10));assert rank==report['spaces'][kind]['rank'];np.testing.assert_allclose(proj,vh[:rank].T@vh[:rank],atol=1e-12)
 cases[channel]={'shards':18,'counts':list(ledger.counts.values()),'raw_to_curvature_to_Richardson_verified':True,'projectors_independent_SVD_verified':True}
# Tiny quotient underflow now fails instead of becoming exact0.
ref=np.ones(10);ref[-1]=0;est=ref.copy();est[-1]=np.nextafter(0.,1.)
try:m._relative(est,ref)
except ValueError:pass
else:raise AssertionError('relative ratio underflow accepted')
# Oversized returned view is rejected before a copy; no huge allocation occurs.
root=P/'response-schema-check';root.mkdir();s=m.ShardStore(root/'arrays');q=m.QueryLedger(root/'progress.json',4);returned=np.broadcast_to(np.ones(1),(1_000_000_000,32));orig=m.np.array
m.np.array=lambda a,*args,**kw:(_ for _ in ()).throw(AssertionError('oversized copied')) if a is returned else orig(a,*args,**kw)
try:
 try:q.query(s,'small',np.ones((4,2)),lambda x:returned,2)
 except ValueError:pass
 else:raise AssertionError('badshape accepted')
finally:m.np.array=orig
assert q.terminal and q.counts['calls_returned']==4 and q.counts['calls_persisted']==0
result={'passed':True,'cases':cases,'relative_ratio_underflow_rejected':True,'oversized_response_precedes_copy':True,'shape_inventory':m.inventory_plan(),'actual_registered_context_calls':0,'registered_RNG_calls':0,'actual_reference_queries':0,'synthetic_callback_batches':8,'synthetic_vector_slots':96,'actual_or_consumed_data_reads':0}
(P/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
