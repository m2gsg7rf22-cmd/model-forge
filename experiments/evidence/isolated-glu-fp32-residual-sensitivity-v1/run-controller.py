from pathlib import Path
import hashlib,json,os,sys,time,subprocess
from importlib.metadata import version
from modelforge.execution import run_json
r=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');source=Path('/Users/adamfaruhi123/.local/share/modelforge-research/experiments/fp32-count-diagnostic-v1-20261025-9e5c51f')
review=json.loads(Path('/private/tmp/modelforge-sensitivity-review/source-review.json').read_text());assert review['approved'] is True
reg=json.loads((r/'experiments/preregistrations/isolated-glu-fp32-residual-sensitivity-v1.json').read_text())
for p,h in dict(review['sources'],**reg['frozen_dependencies']).items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,p
assert not subprocess.check_output(['git','status','--porcelain'],cwd=r,text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()
root=source.parent/('fp32-residual-sensitivity-v1-20261025-'+commit[:7]);root.mkdir(exist_ok=False)
def inventory():return {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()}
before=inventory();report={'source_commit':commit,'reviewed_sources':review['sources'],'frozen_dependencies':reg['frozen_dependencies'],'numpy_version':version('numpy'),'input_inventory_before':before,'private_gate':'closed'}
start=time.monotonic();print(json.dumps({'status':'starting','root':str(root),'optimizer_calls':0,'target_queries':0}),flush=True)
try:
 result=run_json([sys.executable,str(r/'examples/isolated_glu/residual_sensitivity.py')],{'telemetry':str(source/'diagnostic/telemetry.npz'),'config':reg},cwd=root,memory_limit_mb=512,timeout_seconds=180,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1'))
 assert result['process_monitor_scope']=='process_and_children'
 assert result['new_optimizer_calls']==0 and result['target_vector_calls']==0 and not result['accepted_candidate']
 after=inventory();assert before==after,'Source telemetry changed'
 report.update(status='completed',result=result,input_inventory_after=after,input_files_unchanged=True)
except BaseException as exc:
 report.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False,input_inventory_after=inventory())
finally:
 report['controller_seconds']=time.monotonic()-start
 (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'root':str(root),'controller_seconds':report['controller_seconds'],'rows':[{k:v for k,v in row.items() if k!='selected_indices'} for row in report.get('result',{}).get('rows',[])]},indent=2))
if report['status']=='failed':sys.exit(1)
