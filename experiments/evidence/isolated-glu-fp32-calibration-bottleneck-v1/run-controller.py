from pathlib import Path
import hashlib,json,os,sys,time,subprocess
from modelforge.execution import run_json
r=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');base=r.parent/'experiments';old=base/'fp32-fullmap-dev-v1-20261025-8d2d457/public';new=base/'fp32-cubic-richardson-v1-20261025-188a03a'
review=json.loads(Path('/private/tmp/modelforge-bottleneck-review/source-review.json').read_text());assert review['approved'] is True
config=json.loads((r/'experiments/preregistrations/isolated-glu-fp32-calibration-bottleneck-v1.json').read_text())
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==review['controller_sha256']
for p,h in dict(review['sources'],**config['frozen_dependencies']).items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,p
assert not subprocess.check_output(['git','status','--porcelain'],cwd=r,text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip();root=base/('fp32-calibration-bottleneck-v1-20261025-'+commit[:7]);root.mkdir(exist_ok=False)
paths={'multiscale':old/'multiscale.npz','old_manifest':old/'manifest.json','signed':new/'additional-calibration/signed.npz','additional_manifest':new/'additional-calibration/manifest.json','derived':new/'candidate/derived-cubic.npz','candidate':new/'candidate/candidate.npz','fit_report':new/'candidate/fit-report.json'}
start=time.monotonic();report={'source_commit':commit,'reviewed_sources':review['sources'],'frozen_dependencies':config['frozen_dependencies'],'sealed':False,'private_gate':'closed'}
print(json.dumps({'status':'starting','root':str(root),'target_queries':0,'fits':0}),flush=True)
try:
 value=run_json([sys.executable,str(r/'examples/isolated_glu/calibration_bottleneck.py')],{'paths':{k:str(v) for k,v in paths.items()},'output':str(root/'diagnostic'),'config':config},cwd=root,memory_limit_mb=512,timeout_seconds=180,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1'))
 if value['process_monitor_scope']!='process_and_children':raise RuntimeError('Incomplete process-tree monitoring')
 report.update(status='diagnostic_completed',result=value)
except BaseException as exc:
 report.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
finally:
 report['controller_seconds']=time.monotonic()-start
 (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))
if report['status']=='failed':sys.exit(1)
