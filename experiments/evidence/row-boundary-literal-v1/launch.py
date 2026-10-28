from pathlib import Path
import hashlib,json,os,sys,time
r=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');sys.path.insert(0,str(r/'src'))
from modelforge.execution import run_json
b=Path(__file__).parent;env=dict(os.environ);env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONPATH=str(r/'src'))
started=time.monotonic()
scope=run_json([sys.executable,'-c',"import time,json;time.sleep(.15);print(json.dumps({'status':'passed'}))"],{},cwd=b/'scope-engine',memory_limit_mb=512,timeout_seconds=180,output_limit_bytes=2048,env=env)
(b/'scope-receipt.json').write_text(json.dumps(scope,indent=2,sort_keys=True)+'\n')
assert scope['process_monitor_scope']=='process_and_children' and scope['memory_samples']>0
assert not (b/'results.json').exists()
receipt=run_json([sys.executable,str(b/'probe.py')],{},cwd=b/'proof-engine',memory_limit_mb=512,timeout_seconds=180-(time.monotonic()-started),output_limit_bytes=2048,env=env)
assert receipt['process_monitor_scope']=='process_and_children' and receipt['memory_samples']>0
assert receipt['status']=='completed'
receipt['probe_sha256']=hashlib.sha256((b/'probe.py').read_bytes()).hexdigest()
receipt['elapsed_seconds_including_preflight']=time.monotonic()-started
(b/'resource-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt))
