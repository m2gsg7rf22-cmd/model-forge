from pathlib import Path
import hashlib,json,subprocess,os,sys,platform
from importlib.metadata import version
from modelforge.execution import run_json
r=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
review=json.loads(Path('/private/tmp/modelforge-glu-precision-review/source-review.json').read_text())
assert review['approved'] is True
for rel,digest in review['sources'].items():
 assert hashlib.sha256((r/rel).read_bytes()).hexdigest()==digest,rel
assert not subprocess.check_output(['git','status','--porcelain'],cwd=r,text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()
root=Path('/Users/adamfaruhi123/.local/share/modelforge-research/experiments')/('precision-dev-v1-20261025-'+commit[:7]);root.mkdir(exist_ok=False)
print(json.dumps({'status':'starting','root':str(root),'source_commit':commit}),flush=True)
config=json.loads((r/'experiments/preregistrations/isolated-glu-precision-development-v1.json').read_text())
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
try:
 result=run_json([sys.executable,str(r/'examples/isolated_glu/precision.py')],config,cwd=root,memory_limit_mb=512,timeout_seconds=180,env=env)
 assert result['process_monitor_scope']=='process_and_children'
 assert result['vector_calls']==6488064 and len(result['records'])==48
 result['source_commit']=commit;result['reviewed_sources']=review['sources'];result['python']=sys.version;result['platform']=platform.platform();result['numpy_version']=version('numpy')
 (root/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':'completed','root':str(root),'records':48,'vector_calls':result['vector_calls'],'seconds':result['seconds'],'keys':list(result)},indent=2))
except BaseException as exc:
 (root/'failure.json').write_text(json.dumps({'source_commit':commit,'type':type(exc).__name__,'message':str(exc),'automatic_retry':False},indent=2)+'\n')
 raise
