"""Independent integer/schema audit; no NumPy, scientific arrays, RNG or teachers."""
import sys,json,hashlib,ast
from pathlib import Path
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-fresh-inventory-review');sys.path.insert(0,str(R/'examples/isolated_glu'))
import fresh_rank_inventory as m
stores={k:{'shards':0,'payload':0,'upper_arrays':0} for k in ['truth','public','estimator','evaluator']};rawin=rawout=H=0;largest=0
for n,widths in [(4,[2,4,8]),(8,[4,12,20]),(16,[8,24,48]),(32,[16,48,96])]:
 p=n*(n+1)//2;q=min(16*n,p);c=min(q,p-1)
 for w in widths:
  # Independent scalar counts derived directly from the stated mathematical layout.
  expected={'context':('truth',1,10,8*(6*w*n+80*n+2*w*p+16*w)), 'inputs':('truth',32,1,8*4*p*n),'responses':('truth',32,1,8*4*p*n),'reference':('truth',2,1,8*8*n**3),'observations':('public',6,1,8*8*n**3),'bases':('estimator',1,6,8*(3*q*p+3*p)),'choices':('estimator',1,10,8*(3*p*p+13*c+4)),'evaluation':('evaluator',1,11,8*(p+6*w+424))}
  actual={s.kind:s for s in m.profile_schemas(n,w)};assert set(actual)==set(expected)
  for kind,(store,count,fields,payload) in expected.items():
   a=actual[kind];assert (a.store,a.count,len(a.fields),a.payload_bytes)==(store,count,fields,payload)
   bound=payload+4096*fields;assert a.bytes_upper_bound==bound
   stores[store]['shards']+=2*count;stores[store]['payload']+=2*count*payload;stores[store]['upper_arrays']+=2*count*bound;largest=max(largest,bound)
   if kind=='inputs':rawin+=2*count*payload
   if kind=='responses':rawout+=2*count*payload
   if kind in ['reference','observations']:H+=2*count*payload
   # Bound every actual fixed-profile NPY header/ZIP name overhead with plenty of slack.
   for key,shape in a.fields.items():
    header=repr({'descr':'<f8','fortran_order':False,'shape':shape}).encode('ascii');name=(key+'.npy').encode('ascii');npy_header=12+len(header)+64;zip_overhead=30+len(name)+20+46+len(name)+28+22
    assert npy_header+zip_overhead<4096
record=len((json.dumps({'name':'x'*80+'.npz','bytes':8*1024**2,'sha256':'0'*64},separators=(',',':'),sort_keys=True)+'\n').encode());assert record==188
plan=m.inventory_plan();assert plan['vector_calls']==545280 and plan['callback_batches']==768
for name,s in stores.items():
 a=plan['stores'][name];assert a['shards']==s['shards'] and a['payload_bytes']==s['payload'] and a['array_bytes_upper_bound']==s['upper_arrays'];assert a['manifest_bytes_upper_bound']==188*s['shards'];assert a['bytes_upper_bound']==s['upper_arrays']+188*s['shards'];assert 188*s['shards']<(128*1024 if name=='public' else 1024**2)
assert sum(s['shards'] for s in stores.values())==1824 and [stores[k]['shards'] for k in ['truth','public','estimator','evaluator']]==[1608,144,48,24]
assert (rawin,rawout,H)==(119193600,119193600,115015680) and largest==6785056
worst=m.worst_schemas();assert len(worst)==7
preflight=sum(s.bytes_upper_bound for s in worst)+7*188;assert preflight==17591940
combined=sum(s['upper_arrays']+188*s['shards'] for s in stores.values())+preflight+8*1024**2;assert combined==479970708==plan['bytes_upper_bound']<1024**3
for s in m.profile_schemas(32,96):
 if s.kind=='bases':assert s.fields['richardson_basis']==(512,528)
 if s.kind=='context':assert s.fields['atoms']==(192,528)
 if s.kind=='evaluation':assert s.fields['gate_residuals']==(6,96)
# Deliberate constructor boundary controls, no data allocation.
findings={};dims=[1]
try:
 spec=m.ShardSpec('truth','literal',{'x':dims},1);before=spec.payload_bytes;dims[0]=10**9;findings['mutable_shape_accepted_and_exceeds_cap']=spec.payload_bytes>8*1024**2 and before==8
except ValueError:findings['mutable_shape_rejected']=True
for key in ['file','allow_pickle','x'*5000,'x/../bad']:
 try:m.ShardSpec('truth','literal',{key:(1,)},1)
 except ValueError:findings['invalid_key_rejected_'+key[:20]]=True
 else:findings['invalid_key_accepted_'+key[:20]]=True
# Public FD view source must not import privileged inventory.
tree=ast.parse((R/'examples/isolated_glu/fresh_rank_view.py').read_text());assert all(not isinstance(node,ast.ImportFrom) or node.module!='fresh_rank_inventory' for node in ast.walk(tree));assert all(not isinstance(node,ast.Import) or all(x.name!='fresh_rank_inventory' for x in node.names) for node in ast.walk(tree))
result={'fixed_profile_counts_passed':True,'source_sha256':hashlib.sha256((R/'examples/isolated_glu/fresh_rank_inventory.py').read_bytes()).hexdigest(),'stores':stores,'raw_input_bytes':rawin,'raw_output_bytes':rawout,'four_H_views_bytes':H,'largest_npz_upper':largest,'preflight_upper':preflight,'combined_upper':combined,'manifest_record_upper':188,'fixed_profile_header_zip_allowance_checked':True,'constructor_controls':findings,'public_view_no_inventory_import':True,'NumPy_or_scientific_arrays_created':False,'RNG':0,'teacher_or_queries':0,'fitting':0,'actual_resource_protocol_approved':False}
(P/'probe-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
