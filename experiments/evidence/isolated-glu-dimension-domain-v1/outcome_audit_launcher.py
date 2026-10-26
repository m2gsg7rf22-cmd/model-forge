import sys,os,json
from pathlib import Path
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-dimension-domain-outcome-review');sys.path.insert(0,str(R/'src'))
from modelforge.execution import run_json
flags=dict(os.environ);flags.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONPATH=str(R/'src'))
try:
 result=run_json([sys.executable,str(P/'probe.py')],{},cwd=P/'engine',timeout_seconds=180,memory_limit_mb=512,output_limit_bytes=2048,env=flags)
except Exception as e:
 result={'status':'failed','error_type':type(e).__name__,'error':str(e)[:1500]};(P/'audit-resource-receipt.json').write_text(json.dumps(result,indent=2)+'\n');raise
else:
 (P/'audit-resource-receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
