"""Small literal FD-only fixtures; no teacher, RNG, forward, fits or actual data."""
import sys,json,tempfile,hashlib
from pathlib import Path
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-fresh-view-review');sys.path[:0]=[str(R/'src'),str(R/'examples/isolated_glu')]
import fresh_rank_view as m
from scaling_domain_store import ShardStore
base=Path(tempfile.mkdtemp(prefix='literal-',dir=P));checks={}
def setup(label='case'):
 d=base/label;d.mkdir();root=d/'public';root.mkdir();s=ShardStore(root/'arrays');h=np.arange(9*2*3*3,dtype=np.float64).reshape(9,2,3,3);h=h+h.swapaxes(-1,-2);name='c11-fp32'
 for kind in m.VARIANTS:
  s.save(name+'-'+kind+'-h00.npz',hessian=h[:8]);s.save(name+'-'+kind+'-h01.npz',hessian=h[8:])
 meta={'schema':'modelforge-fresh-rank-observations-v1','steps':[.3,.6],'cases':[{'name':name,'n':3,'outputs':2,'probes':9,'compute':'fp32_compute'}]}
 (root/'cases.json').write_text(json.dumps(meta));truth=d/'truth';truth.mkdir();(truth/'context.npz').write_bytes(b'Never read; arbitrary hidden width-dependent size')
 return root,meta,h,truth
def view(root):return m.FDView(root,m.sha(root/'cases.json'),m.sha(root/'arrays/manifest.jsonl'))
def reject(fn,key):
 try:fn()
 except (ValueError,OSError):checks[key]=True
 else:raise AssertionError(key)
root,meta,h,truth=setup();original_open,original_stat,original_iter=Path.open,Path.stat,Path.iterdir
truth_access=[]
def forbidden(p):
 if p==truth or truth in p.parents:truth_access.append(str(p));raise AssertionError('truth filesystem access')
def opened(p,*a,**kw):forbidden(p);return original_open(p,*a,**kw)
def stated(p,*a,**kw):forbidden(p);return original_stat(p,*a,**kw)
def iterd(p,*a,**kw):forbidden(p);return original_iter(p,*a,**kw)
Path.open,Path.stat,Path.iterdir=opened,stated,iterd
try:
 v=view(root);packed=v.packed('c11-fp32');assert not truth_access
finally:Path.open,Path.stat,Path.iterdir=original_open,original_stat,original_iter
# Explicit nested-loop Frobenius packing independently checks sqrt2, row and output order.
expected=np.array([[h[b,o,i,j]*(1. if i==j else np.sqrt(2.)) for i in range(3) for j in range(i,3)] for b in range(9) for o in range(2)])
for a in packed.values():
 assert a.shape==(18,6) and np.array_equal(a,expected) and not a.flags.writeable
 try:a[0,0]=0
 except ValueError:pass
 else:raise AssertionError('writeable')
assert len(v.verify_unchanged()['used_input_sha256'])==6
checks['independent_partial_second_chunk_n3_o2_probe9_packing']=True;checks['no_adjacent_truth_open_stat_or_iter']=True;checks['readonly_public_geometry_no_width']=not hasattr(v.cases[0],'width')
reject(lambda:v.packed('c00-fp64'),'unknown_case_reject')
# Metadata/manifest violations must never reach read_arrays.
orig_read=m.read_arrays;calls=[]
def read_trap(*a,**kw):calls.append(str(a));raise AssertionError('NPZ parsed before metadata guard')
m.read_arrays=read_trap
for field,value in [('width',96),('teacher_type','near'),('seed','secret'),('context_bytes',12345),('n',True),('outputs',33),('probes',17)]:
 r,meta,_,_=setup('bad-'+field);meta['cases'][0][field]=value;(r/'cases.json').write_text(json.dumps(meta));reject(lambda:view(r),'metadata_'+field)
r,meta,_,_=setup('truth-record');mp=r/'arrays/manifest.jsonl';mp.write_text(mp.read_text()+json.dumps({'name':'c11-fp32-context.npz','bytes':96,'sha256':'0'*64})+'\n');reject(lambda:view(r),'truth_size_manifest_reject')
for key,value in [('sha256',None),('sha256','A'*64),('bytes',True)]:
 r,meta,_,_=setup('record-'+str(len(checks)));mp=r/'arrays/manifest.jsonl';records=[json.loads(line) for line in mp.read_text().splitlines()];records[0][key]=value;mp.write_text(''.join(json.dumps(x)+'\n' for x in records));reject(lambda:view(r),'record_'+key+'_'+str(value))
assert not calls;m.read_arrays=orig_read;checks['all_metadata_invalid_before_NPZ_parse']=True
# Hash/header guards, mutation after all six reads, and metadata changes.
r,meta,_,_=setup('header');f=r/'arrays/c11-fp32-fine-h00.npz';np.savez(f,hessian=np.zeros((8,2,3,4)))
mp=r/'arrays/manifest.jsonl';rec=[json.loads(x) for x in mp.read_text().splitlines()];rec[0].update(bytes=f.stat().st_size,sha256=m.sha(f));mp.write_text(''.join(json.dumps(x)+'\n' for x in rec));vv=view(r);reject(lambda:vv.packed('c11-fp32'),'header_shape_reject')
r,_,_,_=setup('mutation');vv=view(r);count=[0]
def mutated(*a,**kw):
 d=orig_read(*a,**kw);count[0]+=1
 if count[0]==6:
  target=r/'arrays/c11-fp32-fine-h00.npz';data=target.read_bytes();target.write_bytes(data[:-1]+bytes([data[-1]^1]))
 return d
m.read_arrays=mutated
try:reject(lambda:vv.packed('c11-fp32'),'same_size_used_mutation_after_read_before_return')
finally:m.read_arrays=orig_read
r,_,_,_=setup('meta-mutation');vv=view(r);(r/'cases.json').write_text((r/'cases.json').read_text()+' ');reject(lambda:vv.packed('c11-fp32'),'metadata_commitment_mutation')
r,_,_,_=setup('null-pin');reject(lambda:m.FDView(r,None,m.sha(r/'arrays/manifest.jsonl')),'external_null_commitment')
result={'passed':True,'checks':checks,'source_sha256':hashlib.sha256((R/'examples/isolated_glu/fresh_rank_view.py').read_bytes()).hexdigest(),'registered_or_actual_data_reads':0,'RNG':0,'teacher_or_forward_queries':0,'fits':0,'private_or941xxx_reads':0,'resource_scope':'Small literal component probes only; no full resource protocol approval'}
(P/'probe-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
