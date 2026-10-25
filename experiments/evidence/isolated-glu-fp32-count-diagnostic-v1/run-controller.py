from pathlib import Path
import hashlib,json,os,sys,time,subprocess
from importlib.metadata import version
from modelforge.execution import run_json
r=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');source=Path('/Users/adamfaruhi123/.local/share/modelforge-research/experiments/fp32-fullmap-dev-v1-20261025-8d2d457')
review=json.loads(Path('/private/tmp/modelforge-count-diagnostic-review/source-review.json').read_text());assert review['approved'] is True
reg=json.loads((r/'experiments/preregistrations/isolated-glu-fp32-count-diagnostic-v1.json').read_text())
for p,h in dict(review['sources'],**reg['frozen_dependencies']).items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,p
assert not subprocess.check_output(['git','status','--porcelain'],cwd=r,text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()
root=source.parent/('fp32-count-diagnostic-v1-20261025-'+commit[:7]);root.mkdir(exist_ok=False)
def inventory():return {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()}
before=inventory();print(json.dumps({'root':str(root),'status':'starting','target_vector_calls':0}),flush=True)
start=time.monotonic();report={'source_commit':commit,'reviewed_sources':review['sources'],'frozen_dependencies':reg['frozen_dependencies'],'numpy_version':version('numpy'),'old_inventory_before':before,'private_gate':'closed'}
try:
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
 result=run_json([sys.executable,str(r/'examples/isolated_glu/count_diagnostic.py')],{'source':str(source),'output':str(root/'diagnostic'),'config':reg},cwd=root,memory_limit_mb=512,timeout_seconds=180,env=env)
 assert result['process_monitor_scope']=='process_and_children'
 assert result['target_vector_calls']==0 and result['profiled_direction_calls']==1
 after=inventory();assert before==after,'Old experiment changed'
 report.update(status='completed',result=result,old_inventory_after=after,old_files_unchanged=True)
except BaseException as exc:
 report.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False,old_inventory_after=inventory())
finally:
 report['controller_seconds']=time.monotonic()-start
 (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))
if report['status']=='failed':sys.exit(1)
