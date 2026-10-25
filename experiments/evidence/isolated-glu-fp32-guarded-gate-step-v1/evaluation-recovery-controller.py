"""One prospective evaluation of a saved audited candidate; never invokes fitter."""
from pathlib import Path
import hashlib,json,os,sys,time,subprocess
from modelforge.execution import run_json
r=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
reviewdir=Path('/private/tmp/modelforge-guarded-gate-step-review')
review=json.loads((reviewdir/'recovery-source-review.json').read_text());assert review['approved'] is True
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==review['controller_sha256']
for p,h in review['sources'].items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,p
original=json.loads((reviewdir/'source-review.json').read_text())
for p,h in original['sources'].items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,p
config=json.loads((r/'experiments/preregistrations/isolated-glu-fp32-guarded-gate-step-v1-evaluation-recovery.json').read_text())
root=Path(config['root']);candidate=root/'candidate'
paths={'failed_report':root/'report.json','fit_report':candidate/'fit-report.json','candidate':candidate/'candidate.npz','baseline':candidate/'baseline.npz','fit_evidence':candidate/'gate-step-evidence.npz','manifest':candidate/'public/manifest.json'}
def inventory():return {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
assert inventory()==config['inputs']
failed=json.loads(paths['failed_report'].read_text());assert failed['status']=='failed' and failed['error']=='Engine result exceeded the output/log budget' and failed['stages']=={}
audit_path=reviewdir/'fit-outcome-audit.json'
assert hashlib.sha256(audit_path.read_bytes()).hexdigest()==review['fit_audit_sha256']
audit=json.loads(audit_path.read_text());assert audit['approved'] is True and audit['candidate_sha256']==config['inputs']['candidate']
assert not (candidate/'evaluation-attempt.json').exists()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=r,text=True).strip()
output=root/'evaluation-recovery';output.mkdir(exist_ok=False)
(output/'attempt.json').write_text(json.dumps({'automatic_retry':False,'new_fits':0,'new_evaluation_calls_reserved':2048,'seed':941032})+'\n')
staged=output/'candidate';staged.mkdir();(staged/'public').mkdir()
full=json.loads(paths['fit_report'].read_text())
compact={k:v for k,v in full.items() if k!='gate_step'}
compact.update(full_fit_report_sha256=config['inputs']['fit_report'],full_fit_details_preserved_externally=True)
encoded=(json.dumps(compact,indent=2,allow_nan=False)+'\n').encode()
assert len(encoded)<2_000_000 and hashlib.sha256(encoded).hexdigest()==config['compact_fit_report_sha256']
(staged/'fit-report.json').write_bytes(encoded)
for key,dest in [('candidate','candidate.npz'),('baseline','baseline.npz'),('manifest','public/manifest.json')]:
 data=paths[key].read_bytes();assert hashlib.sha256(data).hexdigest()==config['inputs'][key]
 (staged/dest).write_bytes(data)
start=time.monotonic();report={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'inputs_before':inventory(),'fit_audit_sha256':review['fit_audit_sha256'],'original_controller_failure_preserved':True,'new_fits':0,'new_direction_solves':0,'sealed':False,'private_gate':'closed','original_build_resource_receipt_unavailable':True}
try:
 value=run_json([sys.executable,str(r/'examples/isolated_glu/guarded_gate_step_evaluate.py')],{'output':str(staged)},cwd=root,memory_limit_mb=512,timeout_seconds=config['remaining_active_controller_budget_seconds'],env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1'))
 assert value['process_monitor_scope']=='process_and_children'
 report.update(status='public_candidate_evaluated',evaluation=value)
except BaseException as exc:
 report.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
finally:
 report['seconds']=time.monotonic()-start;report['inputs_after']=inventory();report['inputs_unchanged']=report['inputs_before']==report['inputs_after']
 if not report['inputs_unchanged']:
  report.update(status='failed',error_type='InputCommitmentChanged',error='Saved fitting artifacts changed',automatic_retry=False)
 report['total_active_controller_seconds']=failed['controller_seconds']+report['seconds']
 (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))
if report['status']=='failed':sys.exit(1)
