from pathlib import Path
import sys,json,hashlib,copy,math
r=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');sys.path[:0]=[str(r/'examples/isolated_glu'),str(r/'src')]
import row_acquisition_controller as c
from row_acquisition_inventory import inventory_plan
p=Path(__file__).parent
checks=[]
# Preserve historical RED evidence; use the same hostile report under isolated IO stubs.
c.public_boundary=lambda *a:None
c.read_json=lambda *a:{'mapping':[{'id':f'c{i:02d}','width':w} for i,w in enumerate((16,48,96))]}
c.exact_store=lambda *a:None
c.records=lambda *a:{f'c{i:02d}-{ch}-context.npz':{'sha256':str(i)*64} for i in range(3) for ch in ('fp64','fp32')}
bad={'cases':[{'case':'duplicate','probes':16,'methods':[{'method':'gap','qualifies':False},{'method':'stable','qualifies':False}]}]*12,'rule_passes':{'gap':True,'stable':True},'by_budget':{'17':{'gap':True,'stable':True}},'paired':[],'certificate':False,'new_queries':0,'new_fits':0,'private_gate':'closed'}
try:c.validate_final(Path('/literal-only-unread'),{}, {},bad)
except ValueError as e:checks.append({'R1_old_counterexample_now_rejected':True,'error':str(e)})
else:raise AssertionError('R1 remains')
# Independently count schema entries/archive field overhead with literal scalar formulas.
n=32;pdim=528;stores={k:{'shards':0,'payload_bytes':0,'array_bytes_upper_bound':0,'shard_limit_bytes':0} for k in ('truth','public','estimator','evaluator')}
def add(store,count,entries,fields):
 e=stores[store];size=8*entries+4096*fields;e['shards']+=count;e['payload_bytes']+=count*entries*8;e['array_bytes_upper_bound']+=count*size;e['shard_limit_bytes']=max(e['shard_limit_bytes'],size);return size
family={}
for m in (16,48,96):
 family['context']=add('truth',2,6*m*n+17*n+64*n+2*m*pdim+17*m,10)
 family['input']=add('truth',68,4*pdim*n,1);family['response']=add('truth',68,4*pdim*n,1)
 family['reference']=add('truth',4,8*n**3,1);family['reference_tail']=add('truth',2,n**3,1)
 add('public',12,8*n**3,1);add('public',6,n**3,1)
 for b in (16,17):
  q=min(b*n,pdim);k=min(q,pdim-1)
  family['bases'+str(b)]=add('estimator',2,3*q*pdim+3*pdim,6)
  family['choices'+str(b)]=add('estimator',2,3*pdim**2+13*k+4,10)
  family['evaluation']=add('evaluator',2,7*pdim+6*m+424,12)
for e in stores.values():
 e['manifest_bytes_upper_bound']=e['shards']*188;e['bytes_upper_bound']=e['array_bytes_upper_bound']+e['manifest_bytes_upper_bound']
plan=inventory_plan();assert stores==plan['stores'],(stores,plan['stores'])
preflight=sum(family.values())+1880
assert preflight==31403888==plan['preflight_bytes_upper_bound']
assert sum(e['shards'] for e in stores.values())==522
assert sum(e['bytes_upper_bound'] for e in stores.values())+preflight+8*1024**2==534980568==plan['bytes_upper_bound']
checks.append({'independent_literal_inventory_matches':True,'stores':stores,'preflight_bytes':preflight,'combined_bytes':534980568})
conf=r/'experiments/preregistrations/paired-row-acquisition-v1.json';v=json.loads(conf.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert set(v['frozen_dependencies'])==c.DEPENDENCIES and len(c.DEPENDENCIES)==24
assert all(sha(r/name)==digest for name,digest in v['frozen_dependencies'].items())
assert v['inventory_plan']==plan and v['policy']==c.POLICY
checks.append({'all24_runtimepins_match':True,'prereg_sha256':sha(conf)})
result={'status':'passed','scope':'literal scalar/schema/source checks only; no RNG/factory/forward/data queries/fitting','checks':checks,'actual_resource_verification':False}
(p/'final-probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
