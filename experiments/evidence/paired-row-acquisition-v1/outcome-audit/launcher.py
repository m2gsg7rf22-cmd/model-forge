import sys,os,json,time
from pathlib import Path
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-row-acquisition-outcome-audit');sys.path.insert(0,str(R/'src'))
from modelforge.execution import run_json
flags=dict(os.environ);flags.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',GIT_OPTIONAL_LOCKS='0',PYTHONPATH=str(R/'src'))
start=time.monotonic()
pre=run_json([sys.executable,'-c',"import time,json;time.sleep(.15);print(json.dumps({'status':'passed','synthetic_scope_preflight':True}))"],{},cwd=P/'scope-engine',memory_limit_mb=512,timeout_seconds=10,output_limit_bytes=2048,env=flags)
(P/'tree-preflight-receipt.json').write_text(json.dumps(pre,indent=2)+'\n')
if pre.get('process_monitor_scope')!='process_and_children' or pre.get('memory_samples',0)<1:raise RuntimeError('TREE preflight unavailable; numeric audit not started')
try:receipt=run_json([sys.executable,str(P/'probe.py')],{},cwd=P/'numeric-engine',memory_limit_mb=512,timeout_seconds=max(.001,180-(time.monotonic()-start)),output_limit_bytes=2048,env=flags)
except Exception as e:receipt={'status':'failed','error_type':type(e).__name__,'error':str(e)[:1000]}
receipt['launcher_elapsed_seconds']=time.monotonic()-start;(P/'audit-resource-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
