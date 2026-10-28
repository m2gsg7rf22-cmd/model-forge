import sys,json,hashlib,tempfile
from pathlib import Path
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-fresh-inventory-review');sys.path[:0]=[str(R/'src'),str(R/'examples/isolated_glu')]
import fresh_rank_inventory as m
from scaling_domain_store import ShardStore
plan=m.inventory_plan();previous=json.loads((P/'source-review.json').read_text())['independent_counts'];root=Path(tempfile.mkdtemp(prefix='empty-stores-',dir=P));maxima={'truth':8*8*32**3+4096,'public':8*8*32**3+4096,'estimator':6785056,'evaluator':8*(528+6*96+424)+4096*11};observed={}
for name,limit in plan['stores'].items():
 assert limit['shard_limit_bytes']==maxima[name]
 old=previous['stores'][name];assert limit['shards']==old['shards'] and limit['payload_bytes']==old['payload'] and limit['array_bytes_upper_bound']==old['upper_arrays']
 assert limit['bytes_upper_bound']==old['upper_arrays']+old['shards']*188
 s=ShardStore(root/name,shard_limit_bytes=limit['shard_limit_bytes'],total_limit_bytes=limit['bytes_upper_bound']);assert s.names==set() and s.manifest_bytes==0 and (s.root/'manifest.jsonl').stat().st_size==0
 observed[name]={'shard_limit_bytes':s.shard_limit_bytes,'total_limit_bytes':s.total_limit_bytes,'empty_manifest_only':True}
assert plan['bytes_upper_bound']==previous['combined_upper']==479970708 and plan['preflight_bytes_upper_bound']==17591940 and plan['scientific_shards']==1824 and plan['vector_calls']==545280 and plan['callback_batches']==768
result={'passed':True,'source_sha256':hashlib.sha256((R/'examples/isolated_glu/fresh_rank_inventory.py').read_bytes()).hexdigest(),'stores':observed,'previous_scientific_counts_and_totals_unchanged':True,'actual_scientific_arrays_allocated':0,'RNG_or_teacher_or_queries_or_fit':0,'actual_runtime_resource_gate_executed':False}
(P/'store-limits-probe-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
