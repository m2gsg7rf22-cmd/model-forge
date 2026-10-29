from pathlib import Path
import hashlib,json,math,re,subprocess,time,sys,platform,importlib.metadata
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');B=R.parent/'experiments';D=B/'joint-response-development-v1-20261029-585eb97';O=Path('/private/tmp/modelforge-joint-response-outcome-audit');COPY=R/'experiments/evidence/joint-response-development-v1/actual'
S='585eb97db907d6a07463e072b1bc690857594bd3';started=time.monotonic();read_bytes=65536;LIMIT=250*1024**2;hashes={};checks=[]
def tick(n=0):
 global read_bytes
 read_bytes+=n
 assert read_bytes<=LIMIT and time.monotonic()-started<=180,'audit budget exceeded'
def safe(p):assert not any(x.is_symlink() for x in (p,*p.parents)) and p.is_file() and '/collection/truth/' not in str(p) and 'modelforge-private' not in str(p)
def data(p,cap=131072):
 safe(p);assert p.stat().st_size<=cap;b=p.read_bytes();tick(len(b));hashes[str(p)]=hashlib.sha256(b).hexdigest();return b
def sha(p,cap=8*1024**2):
 if str(p) in hashes:return hashes[str(p)]
 safe(p);assert p.stat().st_size<=cap;h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024**2):tick(len(b));h.update(b)
 hashes[str(p)]=h.hexdigest();return h.hexdigest()
def unique(pairs):
 d={}
 for k,v in pairs:assert k not in d;d[k]=v
 return d
def decode(b):return json.loads(b,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def obj(p):return decode(data(p))
def lines(p,cap,record,count):
 b=data(p,cap);assert not b or b.endswith(b'\n');rows=b.splitlines();assert len(rows)<=count and all(len(x)+1<=record for x in rows);return [decode(x) for x in rows]
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def native(v,n):return type(v) is int and v==n
def finite(v):return type(v) in (int,float) and math.isfinite(v)
def check(n):checks.append(n)
def git(*a):
 v=subprocess.check_output(['git',*a],cwd=R,timeout=10);tick(len(v));return v
C=obj(D/'controller-report.json');A=obj(D/'outer-monitor.json');I=obj(D/'input-binding.json');T=obj(D/'terminal-binding.json');P=obj(D/'progress.json');F=obj(D/'preflight-receipt.json')
configpath=R/'experiments/preregistrations/joint-response-development-v1.json';config=obj(configpath);reviewpath=R/'experiments/evidence/joint-response-development-v1/source-review/source-review.json';review=obj(reviewpath);markerpath=B/'joint-response-development-v1-attempt.json';M=obj(markerpath)
assert review['full_joint_response_development_protocol_approved'] is True and review['registered_execution_after_gates_approved'] is True and len(review['source_sha256'])==57
for rel,h in review['source_sha256'].items():
 assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and sha(R/rel,2*1024**2)==h
 assert hashlib.sha256(git('show',S+':'+rel)).hexdigest()==h
assert len(config['frozen_dependencies'])==24 and all(review['source_sha256'][f]==h for f,h in config['frozen_dependencies'].items())
assert git('merge-base','--is-ancestor',S,'HEAD')==b''
assert config['runtime']=={'python':platform.python_version(),'numpy':importlib.metadata.version('numpy'),'psutil':importlib.metadata.version('psutil'),'machine':platform.machine(),'executable':sys.executable}
assert C['source_commit']==M['source_commit']==I['source_commit']==T['source_commit']==S and C['config_sha256']==M['config_sha256']==sha(configpath) and C['review_sha256']==M['review_sha256']==sha(reviewpath)
assert C['marker_sha256']==T['marker_sha256']==sha(markerpath) and M['policy']==config['policy'] and M['output']==str(D) and M['preflight_sha256']==sha(D/'preflight-receipt.json')
check('57current_actual_source_pins_24runtime_ancestry_config_approval_marker_binding')
assert C['status']==A['status']=='completed' and A['controller_sha256']==sha(D/'controller-report.json') and A['controller_counters_unknown'] is False and A['resource_receipt_unavailable'] is False
for receipt,status in [(F,'passed'),(A['receipt'],'completed')]:
 assert receipt['status']==status and receipt['process_monitor_scope']=='process_and_children' and type(receipt['memory_samples']) is int and receipt['memory_samples']>0 and type(receipt['peak_sampled_rss_bytes']) is int and 0<receipt['peak_sampled_rss_bytes']<=512*1024**2 and native(receipt['memory_limit_mb'],512)
assert native(C['case_driver_invocations'],6) and native(C['completed_cases'],6) and native(C['numerical_failures'],0) and native(C['abstentions'],0) and native(C['new_queries'],0)
assert C['scientific_reconstruction_success'] is False and C['certificate'] is False and C['private_gate']=='closed' and C['marker_consumed'] is True and C['uncommitted_tail'] is False and C['automatic_retry'] is False
assert C['paired_binding_checks']=={'0':'matched','1':'matched','2':'matched'} and type(C['last_free_bytes']) is int and C['last_free_bytes']>=2*1024**3
assert 0<C['elapsed_before_final_write_seconds']<=A['elapsed_before_outer_write_seconds']<180
check('terminal6case_status_counters_actualTREE_disk_and_before_final_elapsed_receipts')
V=C['view_completion'];assert V['status']=='completed' and V['metadata_sha256']==config['response_metadata'] and V['source_commit']==config['parent_source_commit'] and I['view']['response']['status']=='ready'
assert I['view']['width_is_hypothesis'] is True and native(I['view']['paired_input_commitments'],102) and I['view']['response']['metadata_sha256']==config['response_metadata']
widths=I['widths_hypothesized'];assert widths==I['view']['width_hypotheses']==[96,96,48,48,16,16]
for phase,decodes,metareads in [('validation',408,4),('delivery',408,0),('completion',0,4)]:
 for k,n in [('archives_attempted',408),('archives_read',408),('arrays_validated',decodes),('bytes_read',220703928),('metadata_attempted',metareads),('metadata_read',metareads),('metadata_bytes',180415 if metareads else 0)]:assert native(V[phase][k],n)
assert native(V['delivery']['pairs'],204) and native(V['delivery']['rows'],430848) and native(V['delivery']['payload_bytes'],220594176)
assert set(V['jobs'])=={f'job{i:02d}' for i in range(6)} and all(j['status']=='completed' and native(j['delivered_batches'],34) for j in V['jobs'].values())
assert native(V['new_queries'],0) and native(V['new_fits'],0) and V['certificate'] is False and V['private_gate']=='closed'
check('saved_allowed_response_view_408perpass_1224reads_816decodes_204pairs_430848rows')
CP=[0,*range(256,4000,256),4000];case_summaries=[];archive_hashes={};rootfiles=set();expected_case_files=set()
def cap(name,w):
 state=9*w*32*8+10*4096;params=3*w*32*8+4*4096
 if name in ['initial','selected-physical']:return params
 if name.startswith('checkpoint-'):return state
 if name=='schedule':return 4000*256*8+8192
 if name=='split-indices':return 71808*8+3*4096
 if name.startswith('prediction-'):
  n=int(name[-2:]);return min(8192,71808-n*8192)*32*8+8192
 raise AssertionError(name)
for i,outcome in enumerate(C['cases']):
 job=f'job{i:02d}';root=D/'cases'/job;w=widths[i];assert outcome['job_id']==job and outcome['status']=='completed' and native(outcome['updates_completed'],4000) and outcome['quality_success_label'] is False and native(outcome['predicted_calibration_rows'],71808)
 assert obj(root/'terminal.json')==outcome and T['case_files'][job]
 records=lines(root/'manifest.jsonl',31*4096,4096,31);names=[r['name'] for r in records];expected={'initial','schedule','split-indices','selected-physical'}|{f'checkpoint-{n:04d}' for n in CP}|{f'prediction-{n:02d}' for n in range(9)}
 assert len(names)==len(set(names))==30 and set(names)==expected
 for rec in records:
  name=rec['name'];f=root/(name+'.npz');assert set(rec)=={'name','bytes','sha256'} and type(rec['bytes']) is int and rec['bytes']==f.stat().st_size<=cap(name,w)
  assert sha(f)==rec['sha256']==T['case_files'][job][f.name];archive_hashes[(job,name)]=rec['sha256']
 progress=lines(root/'progress.jsonl',4000*512,512,4000);assert len(progress)==4000
 for n,row in enumerate(progress,1):assert set(row)=={'step','sampled_loss_before_update'} and native(row['step'],n) and finite(row['sampled_loss_before_update']) and row['sampled_loss_before_update']>=0
 journal=lines(root/'checkpoint-metrics.jsonl',17*4096,4096,17);intents=lines(root/'checkpoint-intents.jsonl',17*4096,4096,17);metrics=obj(root/'metrics.json');case=obj(root/'case.json');rng=obj(root/'rng.json')
 assert len(journal)==len(intents)==17 and metrics['checkpoints']==journal and native(metrics['checkpoint_metric_records'],17) and native(metrics['metric_journal_bytes'],(root/'checkpoint-metrics.jsonl').stat().st_size) and native(metrics['updates_completed'],4000)
 selected=None;best=math.inf
 for n,row in enumerate(journal):
  assert native(row['step'],CP[n]) and intents[n]=={'step':CP[n]} and type(intents[n]['step']) is int
  assert set(row)=={'step','train_half_mse','guard_mse','guard_relative_error','selected_at_checkpoint','selected_step_so_far'}
  assert finite(row['train_half_mse']) and row['train_half_mse']>=0 and finite(row['guard_mse']) and row['guard_mse']>=0 and (row['guard_relative_error'] is None or (finite(row['guard_relative_error']) and row['guard_relative_error']>=0))
  chosen=row['guard_mse']<best
  if chosen:selected=row['step'];best=row['guard_mse']
  assert type(row['selected_at_checkpoint']) is bool and row['selected_at_checkpoint'] is chosen and native(row['selected_step_so_far'],selected)
 assert native(metrics['selected_step'],selected) and native(outcome['selected_step'],selected) and metrics['selected_guard_mse']==outcome['selected_guard_mse']==best
 assert metrics['initial']==journal[0] and metrics['final']==journal[-1] and metrics['selected']==journal[CP.index(selected)] and metrics['guard_is_consumed_calibration'] is True
 assert case['job_id']==job and native(case['width_hypothesis'],w) and case['width_is_hypothesis'] is True and case['guard_is_consumed_calibration'] is True and case['guard_centers']==[3,7,11,15] and native(case['train_rows'],54912) and native(case['guard_rows'],16896)
 assert finite(case['sx']) and case['sx']>1e-12 and finite(case['sy']) and case['sy']>1e-12
 states=[]
 for role in ['initial','batches']:
  entry=rng[role];seed=int.from_bytes(hashlib.sha256(f'modelforge-joint-response-development-v1/group{i//2:02d}/{role}'.encode()).digest(),'big');assert type(entry['seed']) is int and entry['seed']==seed==config['paired_seeds'][f'group{i//2:02d}'][role]
  for when in ['before','after']:
   s=entry[when];assert set(s)=={'bit_generator','state','has_uint32','uinteger'} and s['bit_generator']=='PCG64' and set(s['state'])=={'state','inc'} and all(type(v) is int and 0<=v<2**128 for v in s['state'].values()) and s['state']['inc']%2==1 and type(s['has_uint32']) is int and s['has_uint32'] in [0,1] and type(s['uinteger']) is int and 0<=s['uinteger']<2**32;states.append(canon(s).decode())
  assert entry['before']['state']['inc']==entry['after']['state']['inc']
 assert hashlib.sha256(canon(states)).hexdigest()==T['paired_optimizer_bindings'][str(i//2)]['states']
 for file,pin in T['case_files'][job].items():assert sha(root/file)==pin
 metadata={'rng.json','case.json','metrics.json','terminal.json','manifest.jsonl','checkpoint-intents.jsonl','checkpoint-metrics.jsonl','progress.jsonl'}
 assert {p.name for p in root.iterdir()}=={n+'.npz' for n in expected}|metadata
 assert set(T['case_files'][job])=={n+'.npz' for n in expected}|metadata
 assert sum((root/f).stat().st_size for f in metadata-{'progress.jsonl'})<=1024**2
 case_summaries.append({'job_id':job,'width_hypothesis':w,'updates_completed':4000,'selected_step':selected,'initial_train_half_mse':journal[0]['train_half_mse'],'final_train_half_mse':journal[-1]['train_half_mse'],'selected_train_half_mse':metrics['selected']['train_half_mse'],'initial_guard_relative_reported':journal[0]['guard_relative_error'],'selected_guard_relative_reported':metrics['selected']['guard_relative_error'],'final_guard_relative_reported':journal[-1]['guard_relative_error']})
check('all180numeric_archive_hashes_caps_terminals_and6x4000progress_orders_noNPZdecode')
check('all6x17journal_intent_metric_bindings_earliest_min_and_selection_decisions')
paired=[]
for i in [0,2,4]:
 pair={n:archive_hashes[(f'job{i:02d}',n)]==archive_hashes[(f'job{i+1:02d}',n)] for n in ['initial','schedule']};assert all(pair.values());paired.append({'group':i//2,'encoded_initial_schedule_identical':pair})
check('paired_archive_initial_schedule_equality_and_canonical_rng_domain_state_hashes')
assert P['cases']==C['cases'] and native(P['completed_case_visits'],6)
transport=obj(D/'transport/manifest.json');assert transport['memory_exercise'] is True and native(transport['allocated_literal_bytes'],319799296) and native(transport['memory_planning_bytes'],420462592) and sha(D/'transport/manifest.json')==F['manifest_sha256']
assert sum(transport['allocation_plan'].values())==319799296
for rec in transport['records']:
 f=D/'transport'/rec['name'];assert rec['name'] in ['schedule.npz','state.npz'] and native(rec['bytes'],f.stat().st_size) and sha(f)==rec['sha256']
rootnames={'controller-report.json','outer-monitor.json','preflight-receipt.json','input-binding.json','progress.json','terminal-binding.json','cases','transport','preflight-engine'};assert {p.name for p in D.iterdir()}==rootnames and not list((D/'preflight-engine').iterdir()) and not list((D.parent/(D.name+'-launch-engine')).iterdir())
allfiles=[f for f in D.rglob('*') if f.is_file()];total=sum(f.stat().st_size for f in allfiles);assert len(allfiles)==237 and total==186239324 and total+671428083==857667407<=config['inventory']['combined_artifacts_upper_bound']<1024**3
for f in allfiles:
 safe(f)
 if f.suffix=='.json':assert f.stat().st_size<=131072
check('literal_transport_hashes_and_exact_final237file186239324byte_layout_below_caps')
cm=obj(COPY/'copy-manifest.json')
for rel,rec in cm['files'].items():
 f=COPY/rel;assert native(rec['bytes'],f.stat().st_size) and sha(f)==rec['sha256']
 if rel=='terminal-summary.json':continue
 target=markerpath if rel=='attempt-marker.json' else D/rel
 assert sha(target)==rec['sha256']
summary=obj(COPY/'terminal-summary.json');assert summary['source_commit']==S and summary['outer']==A and summary['layout_after_final_outer_write']['retained_bytes']==total and summary['layout_after_final_outer_write']['files']==237
for row in case_summaries:
 x=summary['cases'][row['job_id']];assert x['selected']['step']==row['selected_step'] and x['selected']['guard_relative_error']==row['selected_guard_relative_reported']
check('copied_metadata_manifest_actual_digest_bindings_and_parent_terminal_summary')
result={'approved':True,'saved_engineering_receipt_audit_approved':True,'numeric_metric_audit_approved':False,'actual_source_commit':S,'checks':checks,'source_files_verified':57,'runtime_pins_verified':24,'numeric_archives_stream_hashed':180,'case_summaries':case_summaries,'paired_archive_checks':paired,'output_file_count':237,'retained_postouter_bytes':total,'combined_retained_postouter_bytes':total+671428083,'new_global_marker_bytes_outside_output':markerpath.stat().st_size,'reported_outer_resources':A['receipt'],'reported_preflight_resources':F,'controller_elapsed_before_final_write_seconds':C['elapsed_before_final_write_seconds'],'outer_elapsed_before_final_write_seconds':A['elapsed_before_outer_write_seconds'],'audit_read_bytes_including64KiBinspection_allowance':read_bytes,'audit_wall_seconds':time.monotonic()-started,'audit_resource_measurement':False,'audit_libraries':'stdlib only; git read-only source commands; no numerical/process workers','original_observation_reads':0,'numeric_npz_decodes':0,'rng_regenerations':0,'optimizer_replays':0,'new_queries':0,'certificate':False,'private_gate':'closed','scientific_reconstruction_success':False,'artifact_sha256':hashes,'limitations':['Engineering receipts, source and streamed archive hashes only. No numeric NPZ decoding, candidate forward, independent metric recomputation, gradient/optimizer or RNG replay. Initial/schedule encoded equality verifies saved byte correspondence, not independent seed-to-draw correctness.','All selected guard relative errors reported exceed1, with jobs02-05 selecting initialization; this is negative consumed-calibration evidence, not an independently recomputed quality conclusion or unseen generalization test. Width rank/2 remains a hypothesis.','Source/current+actual commit pins and receipts support declared chronology; every original read/update event is not independently proven. Original observations/teacher/context/reference/seed ledger/heldout/private assets were not read.','Original RSS is sampled, not a continuous peak/OS cap. Controller and outer elapsed are before final writes. This small audit did not measure its own RSS.','Global marker is separately hashed and sized outside the output-root retained count. No new experiment, promotion, fresh evaluation or private-model authorization follows.']}
(O/'outcome-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(O/'outcome-audit.md').write_text('אושר אודיט הקבלות ההנדסיות בלבד: 57 חתימות מקור/24 runtime, marker, מניפסטים והעותקים הקטנים תואמים; 180 ארכיוני עקבות גובבו בזרימה ללא פענוח NPZ, ולכל ששת המקרים אומתו 4000 רשומות progress ו־17 checkpoints/כוונות/מטריקות ובחירת minimum מוקדם. initial/schedule הזוגיים זהים ברמת הארכיון וחתימות states תואמות. הפלט הסופי כולל 237 קבצים ו־186239324 בתים (יחד עם הקודם857667407); outer מדווח TREE420052992 בתים/197 דגימות, וזמן12.448610334 שניות לפני כתיבה סופית. שגיאות guard הנבחרות המדווחות כולן מעל1, ובמקרים02–05 נבחרה אתחול0; אין מכך הצלחת שחזור. האודיט לא פענח מערכים ולא חישב candidate forward/מטריקות עצמאיות, לא קרא תצפיות מקור/אמת מורה/heldout/private ולא שיחזר RNG או אופטימיזציה. RSS מדוגם והקבלות אינן הוכחת כל אירוע; guard הוא כיול נצרך והרוחב השערה. השער הפרטי סגור ואין אישור מחקר נוסף.\n')
copy=['audit.py','outcome-audit.json','outcome-audit.md'];(O/'copy-manifest.json').write_text(json.dumps({'files':{n:{'sha256':hashlib.sha256((O/n).read_bytes()).hexdigest(),'bytes':(O/n).stat().st_size} for n in copy},'arrays_included':False},indent=2)+'\n')
print(json.dumps({'approved':True,'checks':len(checks),'read_bytes':read_bytes,'wall_seconds':time.monotonic()-started,'selected_steps':[x['selected_step'] for x in case_summaries],'hashes':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['outcome-audit.json','outcome-audit.md','copy-manifest.json']}},indent=2))
