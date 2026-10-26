import importlib.util,json
from pathlib import Path
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-domain-store-review');s=importlib.util.spec_from_file_location('ind_store_draft',R/'examples/isolated_glu/scaling_domain_store.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
store=m.ShardStore(P/'draft-size-before-finite',shard_limit_bytes=512,total_limit_bytes=1024)
a=np.broadcast_to(np.array([1.]),(1024,))
original=m.np.isfinite;called=[]
def forbidden(x):called.append(True);raise RuntimeError('finitecheck reached before payload guard')
m.np.isfinite=forbidden
try:store.save('oversized.npz',data=a)
except RuntimeError:pass
else:raise AssertionError('expected current unsafe finite-first order')
finally:m.np.isfinite=original
assert called and not (store.root/'oversized.npz').exists()
store2=m.ShardStore(P/'draft-duplicate-key');item=store2.save('one.npz',data=np.arange(4,dtype=float))
line=json.dumps(item);line=line.replace('{','{"name":"discarded-conflicting-name.npz",',1);(store2.root/'manifest.jsonl').write_text(line+'\n');accepted=m.verify_manifest(store2.root)
assert accepted['shards']==1
result={'finite_checked_before_oversized_payload_rejection':True,'conflicting_duplicate_json_key_accepted':True,'registered_seeds_or_data':False,'rng_calls':0,'queries':0}
(P/'draft-probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
