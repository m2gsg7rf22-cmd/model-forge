"""Literal/view/arange transport probes only; no RNG or scientific inputs."""
import importlib.util,json,hashlib
from pathlib import Path
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-domain-store-review');sp=importlib.util.spec_from_file_location('ind_store_final',R/'examples/isolated_glu/scaling_domain_store.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
# Safe finite-scan sentinel: oversized strided/broadcast payload must not be scanned.
s=m.ShardStore(P/'final-oversized',shard_limit_bytes=512,total_limit_bytes=2048);large=np.broadcast_to(np.array([1.]),(1_000_000_000,));original=m.np.isfinite
m.np.isfinite=lambda a:(_ for _ in ()).throw(AssertionError('oversized finite allocation'))
try:
 try:s.save('large.npz',data=large)
 except ValueError:pass
 else:raise AssertionError('oversized accepted')
finally:m.np.isfinite=original
assert s.bytes_reserved==0 and not (s.root/'large.npz').exists()
# Serialized NPZ overhead must be included, even when native payload fits.
s2=m.ShardStore(P/'final-serialized',shard_limit_bytes=512,total_limit_bytes=2048)
try:s2.save('header-overhead.npz',data=np.zeros(32))
except ValueError as e:assert 'Serialized' in str(e)
else:raise AssertionError('serialized overhead ignored')
assert s2.bytes_reserved==0 and not (s2.root/'header-overhead.npz').exists()
# Independent representative largest basis shape; all values arange, no teacher.
s3=m.ShardStore(P/'final-basis-shape');a=np.arange(528*528,dtype=np.float64).reshape(528,528);item=s3.save('basis.npz',basis=a,singular_values=np.arange(528,dtype=np.float64))
with np.load(s3.root/'basis.npz',allow_pickle=False) as z:assert z['basis'].tobytes()==a.tobytes() and set(z.files)=={'basis','singular_values'}
orig_read=Path.read_bytes
def no_shard_readall(path):
 if path.suffix=='.npz':raise AssertionError('unbounded raw NPZ read')
 return orig_read(path)
Path.read_bytes=no_shard_readall
try:verified=m.verify_manifest(s3.root)
finally:Path.read_bytes=orig_read
assert verified['bytes_reserved']==s3.bytes_reserved and verified['largest_shard_bytes']<=m.SHARD_LIMIT
# Conflicting duplicate JSON keys must be malformed, not last-wins.
s4=m.ShardStore(P/'final-duplicates');item4=s4.save('one.npz',data=np.arange(3,dtype=float));line=json.dumps(item4).replace('{','{"name":"discarded.npz",',1);(s4.root/'manifest.jsonl').write_text(line+'\n')
try:m.verify_manifest(s4.root)
except ValueError:pass
else:raise AssertionError('duplicate JSON name accepted')
# A short manifest append is terminal even when the full shard was already saved.
s5=m.ShardStore(P/'final-partial-append');orig_open=Path.open
class Short:
 def __init__(self,f):self.f=f
 def __enter__(self):return self
 def __exit__(self,*a):self.f.close()
 def write(self,b):return self.f.write(b[:5])
def patched(path,mode='r',*args,**kw):
 f=orig_open(path,mode,*args,**kw)
 return Short(f) if path==s5.root/'manifest.jsonl' and mode=='ab' else f
Path.open=patched
try:
 try:s5.save('partial.npz',data=np.ones(3))
 except OSError:pass
 else:raise AssertionError('short append accepted')
finally:Path.open=orig_open
assert s5.terminal and s5.bytes_reserved>0 and 'partial.npz' in s5.names and (s5.root/'partial.npz').exists()
try:s5.save('retry.npz',data=np.ones(1))
except ValueError:pass
else:raise AssertionError('terminal retry allowed')
try:m.verify_manifest(s5.root)
except ValueError:pass
else:raise AssertionError('partial manifest verified')
# Preexisting manifest symlink cannot redirect a subsequent append.
s6=m.ShardStore(P/'final-manifest-link');outside=P/'outside-sentinel';outside.write_bytes(b'unchanged');(s6.root/'manifest.jsonl').unlink();(s6.root/'manifest.jsonl').symlink_to(outside)
try:s6.save('blocked.npz',data=np.ones(1))
except ValueError:pass
else:raise AssertionError('manifest symlink append accepted')
assert outside.read_bytes()==b'unchanged' and not (s6.root/'blocked.npz').exists()
result={'passed':True,'oversized_payload_precedes_finite_scan':True,'serialized_npz_overhead_rejected_before_disk':True,'profile_basis_528_shape_roundtrip':True,'profile_shard_serialized_bytes':verified['largest_shard_bytes'],'streaming_verifier_no_npz_readall':True,'duplicate_json_keys_rejected':True,'short_manifest_append_terminal_reservation_preserved':True,'manifest_symlink_append_blocked':True,'registered_teachers':0,'rng_calls':0,'reference_queries':0,'actual_or_consumed_data_reads':0,'source_sha256':hashlib.sha256((R/'examples/isolated_glu/scaling_domain_store.py').read_bytes()).hexdigest()}
(P/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
