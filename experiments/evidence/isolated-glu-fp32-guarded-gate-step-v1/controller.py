from pathlib import Path
import hashlib,json,os,sys,time,subprocess
from importlib.metadata import version
from modelforge.execution import run_json
r=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');base=Path('/Users/adamfaruhi123/.local/share/modelforge-research/experiments');oldcal=base/'fp32-fullmap-dev-v1-20261025-8d2d457';oldtelemetry=base/'fp32-count-diagnostic-v1-20261025-9e5c51f'
review=json.loads(Path('/private/tmp/modelforge-guarded-gate-step-review/source-review.json').read_text());assert review['approved'] is True
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==review['controller_sha256']
config=json.loads((r/'experiments/preregistrations/isolated-glu-fp32-guarded-gate-step-v1.json').read_text())
for p,h in dict(review['sources'],**config['frozen_dependencies']).items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,p
assert not subprocess.check_output(['git','status','--porcelain'],cwd=r,text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip();root=base/('fp32-guarded-gate-step-v1-20261025-'+commit[:7]);root.mkdir(exist_ok=False)
def inventory():return {d.name+'/'+str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest() for d in [oldcal,oldtelemetry] for p in sorted(d.rglob('*')) if p.is_file()}
new=base/'fp32-guarded-forward-convergence-v1-20261025-4da51d4/candidate'
cal=base/'fp32-guarded-forward-polish-v1-20261025-db8fb33/calibration'
paths={'baseline':new/'candidate.npz','fit_report':new/'fit-report.json','old_manifest':oldcal/'public/manifest.json','forward_manifest':cal/'manifest.json','observations':cal/'observations.npz'}
start=time.monotonic();before=inventory();report={'source_commit':commit,'reviewed_sources':review['sources'],'frozen_dependencies':config['frozen_dependencies'],'numpy_version':version('numpy'),'input_inventory_before':before,'stages':{},'private_gate':'closed','sealed':False}
print(json.dumps({'status':'starting','root':str(root),'new_calibration_calls':0}),flush=True)
stage='build';env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
try:
 for stage in ['build','evaluate']:
  if stage=='evaluate' and not report['stages']['build']['candidate_built']:
   report['status']='hypothesis_rejected';break
  remaining=180-(time.monotonic()-start)
  if remaining<=0:raise TimeoutError('Overall180sbudget exhausted')
  script='guarded_gate_step.py' if stage=='build' else 'guarded_gate_step_evaluate.py'
  payload={'paths':{k:str(v) for k,v in paths.items()},'output':str(root/'candidate'),'config':config} if stage=='build' else {'output':str(root/'candidate')}
  value=run_json([sys.executable,str(r/'examples/isolated_glu'/script)],payload,cwd=root,memory_limit_mb=512,timeout_seconds=remaining,env=env)
  if value['process_monitor_scope']!='process_and_children':raise RuntimeError('Incomplete process-tree monitoring')
  report['stages'][stage]=value;(root/(stage+'-receipt.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
 else:report['status']='public_candidate_evaluated'
 after=inventory();assert before==after,'Old experiment changed';report.update(input_inventory_after=after,old_inputs_unchanged=True)
except BaseException as exc:
 report.update(status='failed',failed_stage=stage,error_type=type(exc).__name__,error=str(exc),automatic_retry=False,input_inventory_after=inventory())
finally:
 report['controller_seconds']=time.monotonic()-start
 report['max_worker_tree_sampled_rss_bytes']=max([v['peak_sampled_rss_bytes'] for v in report['stages'].values()] or [0])
 (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'root':str(root),'seconds':report['controller_seconds'],'worker_peak':report['max_worker_tree_sampled_rss_bytes'],'candidate_built':report['stages'].get('build',{}).get('candidate_built'),'evaluation':report['stages'].get('evaluate')},indent=2))
if report['status']=='failed':sys.exit(1)
