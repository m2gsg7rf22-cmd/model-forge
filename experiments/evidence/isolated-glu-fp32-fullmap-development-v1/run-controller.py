from pathlib import Path
import hashlib,json,os,sys,time,subprocess
from importlib.metadata import version
from modelforge.execution import run_json
r=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
review=json.loads(Path('/private/tmp/modelforge-fp32-fullmap-review/source-review.json').read_text())
assert review['approved'] is True
reg=json.loads((r/'experiments/preregistrations/isolated-glu-fp32-fullmap-development-v1.json').read_text())
for p,h in dict(review['sources'],**reg['frozen_dependencies']).items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,p
assert not subprocess.check_output(['git','status','--porcelain'],cwd=r,text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()
root=Path('/Users/adamfaruhi123/.local/share/modelforge-research/experiments')/('fp32-fullmap-dev-v1-20261025-'+commit[:7]);root.mkdir(exist_ok=False)
report={'source_commit':commit,'reviewed_sources':review['sources'],'frozen_dependencies':reg['frozen_dependencies'],'numpy_version':version('numpy'),'stages':{},'status':'starting','private_gate':'closed','sealed':False}
print(json.dumps({'status':'starting','root':str(root),'commit':commit}),flush=True)
start=time.monotonic();stage='prepare'
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
try:
 for stage in ['prepare','fit','evaluate']:
  if stage=='evaluate' and not report['stages']['fit']['candidate_built']:
   report['status']='candidate_rejected';break
  remaining=180-(time.monotonic()-start)
  if remaining<=0:raise TimeoutError('Overall180sbudget exhausted')
  value=run_json([sys.executable,str(r/'examples/isolated_glu/fp32_fullmap.py')],{'stage':stage,'root':str(root)},cwd=root,memory_limit_mb=512,timeout_seconds=remaining,env=env)
  if value['process_monitor_scope']!='process_and_children':raise RuntimeError('Incomplete process-tree monitoring')
  report['stages'][stage]=value
  (root/(stage+'-receipt.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
 else:report['status']='public_candidate_evaluated'
except BaseException as exc:
 report.update(status='failed',failed_stage=stage,error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
finally:
 report['controller_seconds']=time.monotonic()-start
 report['max_stage_peak_sampled_rss_bytes']=max([v['peak_sampled_rss_bytes'] for v in report['stages'].values()] or [0])
 (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))
if report['status']=='failed':sys.exit(1)
