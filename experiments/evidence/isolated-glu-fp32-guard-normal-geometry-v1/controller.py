"""One bounded read-only calibration diagnostic; never fits or evaluates a model."""
from pathlib import Path
import hashlib,json,os,sys,time,subprocess
from modelforge.execution import run_json
r=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');base=r.parent/'experiments'
review=json.loads(Path('/private/tmp/modelforge-guard-normal-geometry-review/source-review.json').read_text());assert review['approved'] is True
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==review['controller_sha256']
config=json.loads((r/'experiments/preregistrations/isolated-glu-fp32-guard-normal-geometry-v1.json').read_text())
for name,h in dict(review['sources'],**config['frozen_dependencies']).items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
assert not subprocess.check_output(['git','status','--porcelain'],cwd=r,text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip();root=base/('fp32-guard-normal-geometry-v1-20261025-'+commit[:7]);root.mkdir(exist_ok=False)
old=base/'fp32-guarded-gate-step-v1-20261025-3d3b622';cal=base/'fp32-guarded-forward-polish-v1-20261025-db8fb33/calibration'
paths={'candidate':old/'candidate/candidate.npz','compact_metadata':old/'evaluation-recovery/candidate/fit-report.json','full_fit_report':old/'candidate/fit-report.json','observations':cal/'observations.npz','manifest':cal/'manifest.json','original_manifest':base/'fp32-fullmap-dev-v1-20261025-8d2d457/public/manifest.json'}
def inventory():return {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
start=time.monotonic();before=inventory();report={'source_commit':commit,'reviewed_sources':review['sources'],'frozen_dependencies':config['frozen_dependencies'],'inputs_before':before,'status':'starting','direction_solves':0,'target_queries':0,'candidate_updates':0,'private_gate':'closed'}
print(json.dumps({'status':'starting','root':str(root),'direction_solves':0,'target_queries':0}),flush=True)
try:
 assert before==config['inputs'],'Input commitments changed before execution'
 remaining=180-(time.monotonic()-start)
 if remaining<=0:raise TimeoutError('Overall180sbudget exhausted')
 value=run_json([sys.executable,str(r/'examples/isolated_glu/guard_normal_geometry.py')],{'paths':{k:str(p) for k,p in paths.items()},'output':str(root/'diagnostic'),'config':config},cwd=root,memory_limit_mb=512,timeout_seconds=remaining,output_limit_bytes=2048,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1'))
 report['diagnostic']=value
 assert value['process_monitor_scope']=='process_and_children'
 report['status']='diagnostic_completed'
except BaseException as exc:
 report.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
finally:
 report['inputs_after']=inventory();report['inputs_unchanged']=before==report['inputs_after'];report['seconds']=time.monotonic()-start
 if not report['inputs_unchanged']:report.update(status='failed',error_type='InputCommitmentChanged',error='Read-only input changed',automatic_retry=False)
 (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'root':str(root),'seconds':report['seconds'],'diagnostic':report.get('diagnostic'),'error':report.get('error')},indent=2))
if report['status']=='failed':sys.exit(1)
