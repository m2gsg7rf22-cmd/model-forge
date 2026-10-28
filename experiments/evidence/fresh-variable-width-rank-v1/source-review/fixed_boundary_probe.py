import sys,json,tempfile,hashlib
from pathlib import Path
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-fresh-protocol-review');sys.path[:0]=[str(R/'src'),str(R/'examples/isolated_glu')]
import fresh_rank_evaluation as e
import fresh_rank_controller as c
import fresh_rank_protocol_inventory as inv
checks={}
a=np.diag([1.,0.]);b=np.array([[1.,1e-200]]);comp=e.component_residuals(e.reference_summary(a),b);assert comp[0]>0 and abs(comp[0]/1e-200-1)<1e-15;assert e.energy_loss(a,b)==comp[0];checks['representable_tiny_component_preserved']=True
try:e.component_residuals(e.reference_summary(np.diag([1.,1e-200,0.])),np.array([[1.,0.,0.],[0.,1.,1e-200]]))
except ValueError:checks['weighted_underflow_fail_closed']=True
else:raise AssertionError('weighted underflow accepted')
for a,b in [(np.diag([3.,4.]),np.array([[1.,0.]])),(np.array([[3.,4.,0.]]),np.array([[1.,0.,0.]])),(np.eye(2),np.array([[.6,.8]]))]:
 comp=e.component_residuals(e.reference_summary(a),b);assert abs(np.sqrt(np.dot(comp,comp))-e.energy_loss(a,b))<2e-15
 if a.shape==(1,3):assert comp.shape==(3,) and np.all(comp[1:]==0)
checks['independent_energy_identity_and_rowlimited_padding']=True
plan=inv.inventory_plan();assert plan['bytes_upper_bound']==480302932 and plan['preflight_bytes_upper_bound']==17621380 and plan['stores']['evaluator']['bytes_upper_bound']==1531968 and plan['stores']['evaluator']['shard_limit_bytes']==86720;assert plan['scientific_shards']==1824 and plan['vector_calls']==545280 and plan['callback_batches']==768
checks['protocol_inventory_extension_counts_and_caps']=True
# Isolated layout: five tiny literal archive records, exact metadata, empty engine dirs.
root=Path(tempfile.mkdtemp(prefix='exact-layout-',dir=P))
for name in c.METADATA_PATHS:
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
for directory in c.STORE_PATHS.values():
 p=root/directory;p.mkdir(parents=True,exist_ok=True);(p/'literal.npz').write_bytes(b'literal');(p/'manifest.jsonl').write_text(json.dumps({'name':'literal.npz','bytes':7,'sha256':hashlib.sha256(b'literal').hexdigest()})+'\n')
for stage in ['scope','transport','collection','estimation','evaluation']:(root/(stage+'-engine')).mkdir()
c.exact_layout(root)
for name,isdir in [('extra-uncommitted.npz',False),('extra-metadata.json',False),('unexpected-directory',True),('scope-engine/extra.txt',False)]:
 p=root/name
 if isdir:p.mkdir()
 else:p.write_bytes(b'literal')
 try:c.exact_layout(root)
 except ValueError:checks['layout_reject_'+name]=True
 else:raise AssertionError('layout accepted '+name)
 if isdir:p.rmdir()
 else:p.unlink()
c.exact_layout(root)
result={'passed':True,'checks':checks,'sources_sha256':{n:hashlib.sha256((R/'examples/isolated_glu'/n).read_bytes()).hexdigest() for n in ['fresh_rank_evaluation.py','fresh_rank_controller.py','fresh_rank_protocol_inventory.py']},'actual_owned_or_private_or941xxx_reads':0,'RNG_or_teacher_or_target_queries_or_fit':0,'subprocesses_launched':0,'full_protocol_approved':False}
(P/'fixed-boundary-probe-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
