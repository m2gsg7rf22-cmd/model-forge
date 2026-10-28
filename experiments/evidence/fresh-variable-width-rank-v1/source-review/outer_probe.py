"""Fake-engine outer monitor checks; no controlled_run or actual subprocess."""
import sys,json,tempfile,hashlib
from pathlib import Path
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-fresh-protocol-review');sys.path[:0]=[str(R/'src'),str(R/'examples/isolated_glu')]
import fresh_rank_controller as c
base=Path(tempfile.mkdtemp(prefix='outer-literal-',dir=P));checks={};details={}
for mode in ['complete','scope','kill','innerfailed','one_shot_receipt_write_failure','existing']:
 output=base/mode;request={'repo':str(base/'repository'),'output':str(output),'config':'literal-config','review':'literal-review'};original=c.save_json;calls=[];writes=[0]
 if mode=='existing':output.mkdir();(output/'outer-monitor.json').write_text('original')
 def save(path,value):
  if path.name=='outer-monitor.json' and mode=='one_shot_receipt_write_failure':
   writes[0]+=1
   if writes[0]==1:raise OSError('synthetic first receipt write failure')
  return original(path,value)
 c.save_json=save
 def engine(command,payload,**kw):
  calls.append(payload);assert payload['operation']=='controlled_run' and kw['memory_limit_mb']==512 and kw['timeout_seconds']==180 and kw['output_limit_bytes']==2048
  assert all(kw['env'][x]=='1' for x in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'])
  output.mkdir();c.save_json(output/'controller-report.json',{'status':'failed' if mode=='innerfailed' else 'completed'})
  if mode=='kill':raise TimeoutError('synthetic killed controller')
  return {'status':'completed','process_monitor_scope':'process_only_children_unavailable' if mode=='scope' else 'process_and_children','memory_samples':1,'peak_sampled_rss_bytes':123,'memory_limit_mb':512}
 try:
  try:v=c.launch(request,engine=engine);raised=False
  except c.ProtocolError:raised=True
  if mode=='existing':assert raised and not calls and (output/'outer-monitor.json').read_text()=='original';checks[mode]=True;continue
  receipt=json.loads((output/'outer-monitor.json').read_text());details[mode]={'raised':raised,'persisted_status':receipt['status'],'error_type':receipt.get('error_type')}
  if mode=='complete':assert not raised and receipt['status']=='completed' and receipt['controller_sha256']==c.sha(output/'controller-report.json');checks[mode]=True
  else:checks[mode]=raised and receipt['status']=='failed'
 finally:c.save_json=original
result={'all_passed':all(checks.values()),'checks':checks,'details':details,'controller_sha256':hashlib.sha256((R/'examples/isolated_glu/fresh_rank_controller.py').read_bytes()).hexdigest(),'actual_subprocess_or_factory_RNG_data_queries_fits':0,'full_protocol_approved':False}
(P/'outer-probe-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
