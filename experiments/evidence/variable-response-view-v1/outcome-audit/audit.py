from pathlib import Path
import hashlib,json,re,subprocess,platform,importlib.metadata,time
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
B=R.parent/'experiments';D=B/'variable-response-view-v1-20261029-ff11672';O=Path('/private/tmp/modelforge-variable-response-outcome-audit')
S='ff11672240d05625aec014b2840d64397f47fae1';started=time.monotonic();commitments={};checks=[]
def blob(p,cap=2097152):
 assert '/collection/truth/' not in str(p) and '/modelforge-private' not in str(p)
 assert not any(x.is_symlink() for x in (p,*p.parents)) and p.is_file() and p.stat().st_size<=cap
 b=p.read_bytes();assert len(b)<=cap;commitments[str(p)]=hashlib.sha256(b).hexdigest();return b
def sha(p,cap=2097152):return hashlib.sha256(blob(p,cap)).hexdigest()
def obj(p,cap=2097152):
 def unique(pairs):
  d={}
  for k,v in pairs:assert k not in d;d[k]=v
  return d
 return json.loads(blob(p,cap),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def check(name,condition=True):assert condition,name;checks.append(name)
def git(*args):return subprocess.check_output(['git',*args],cwd=R,timeout=10)
def native(v,n):return type(v) is int and v==n
def tree(v,status):
 assert v['status']==status and v['process_monitor_scope']=='process_and_children'
 assert type(v['memory_samples']) is int and v['memory_samples']>0
 assert type(v['peak_sampled_rss_bytes']) is int and 0<v['peak_sampled_rss_bytes']<=512*1024**2 and native(v['memory_limit_mb'],512)
C=obj(D/'controller-report.json');A=obj(D/'outer-monitor.json');V=obj(D/'view-report.json');P=obj(D/'progress.json');seal=obj(D/'descriptor-seal.json')
markerpath=B/'variable-response-view-v1-attempt.json';M=obj(markerpath)
configpath=R/'experiments/preregistrations/variable-response-view-v1.json';config=obj(configpath)
reviewpath=R/'experiments/evidence/variable-response-view-v1/source-review/source-review.json'
review=obj(reviewpath)
assert review['full_variable_response_read_protocol_approved'] is True and len(review['source_sha256'])==21
for name,h in review['source_sha256'].items():
 assert not Path(name).is_absolute() and '..' not in Path(name).parts
 assert sha(R/name)==h and hashlib.sha256(git('show',S+':'+name)).hexdigest()==h
check('all21_current_and_actual_commit_source_pins')
assert len(config['frozen_dependencies'])==7
for f,h in config['frozen_dependencies'].items():assert review['source_sha256'][f]==h
assert config['runtime']=={'python':platform.python_version(),'numpy':importlib.metadata.version('numpy'),'psutil':importlib.metadata.version('psutil')}
assert git('merge-base','--is-ancestor',S,'HEAD')==b''
assert C['source_commit']==M['source_commit']==S
assert M['config_sha256']==sha(configpath) and M['review_sha256']==sha(reviewpath)
assert M['descriptor_sha256']==sha(D/'descriptor-seal.json') and C['marker_sha256']==sha(markerpath)
assert M['policy']==config['policy'] and M['output']==str(D) and M['id']==config['id']
check('source_ancestry_runtime_config_review_marker_descriptor_binding')
base=R/'experiments/evidence/paired-row-acquisition-v1'
publicpaths={'audit':base/'outcome-audit/outcome-audit.json','commitments':base/'outcome-audit/artifact-commitments.json','collection':base/'actual/collection/collection-report.json','progress':base/'actual/collection/query-progress.json'}
for n,p in publicpaths.items():assert sha(p)==config['public_metadata'][n]
K=obj(publicpaths['commitments']);prior=obj(publicpaths['audit']);coll=obj(publicpaths['collection']);prog=obj(publicpaths['progress'])
assert prior['approved'] is True and prior['saved_outcome_audit_approved'] is True
assert prior['actual_source_commit']==coll['source_commit']==config['parent_source']
assert prior['artifact_commitments_sha256']==sha(publicpaths['commitments'])
for n in ['collection','progress']:
 leaf='collection-report.json' if n=='collection' else 'query-progress.json'
 assert K['metadata_sha256']['collection/'+leaf]==sha(publicpaths[n])
for c in [coll,prog]:
 for k,n in [('calls_reserved',430848),('calls_attempted',430848),('calls_returned',430848),('calls_persisted',430848),('callback_batches',204)]:assert native(c[k],n)
check('static_public_parent_lineage_and_query_receipts_no_raw_source_reads')
assert set(seal)=={'pairs','public_metadata'} and seal['public_metadata']==config['public_metadata'] and len(seal['pairs'])==204
original=str(B/'paired-row-acquisition-v1-20261028-07235ac/collection')
for i,p in enumerate(seal['pairs']):
 job=i//34;step=(i%34)//17;probe=i%17;case=f'c{job//2:02d}-'+('fp64' if job%2==0 else 'fp32')
 assert p['job_id']==f'job{job:02d}' and native(p['index'],i%34) and set(p)=={'job_id','index','inputs','responses'}
 for field in ['inputs','responses']:
  x=p[field];name=f'{case}-s{step}-x{probe:02d}-{field}.npz'
  assert set(x)=={'name','field','shape','digest'} and x['name']==name and x['field']==field and x['shape']==[2112,32]
  assert all(type(n) is int for n in x['shape']) and x['digest']==K['artifact_sha256'][original+'/truth/'+name]
check('408_exact_ordered_descriptors_match_public_archive_commitments')
chainbytes=blob(D/'delivery.jsonl',131072);head='0'*64;lines=chainbytes.splitlines();assert len(lines)==204
for i,line in enumerate(lines):
 x=json.loads(line);claimed=x.pop('chain');assert len(line)+1<=512
 assert set(x)=={'job_id','index','inputs_shape','responses_shape','inputs_sha256','responses_sha256','previous'}
 assert x['job_id']==f'job{i//34:02d}' and native(x['index'],i%34) and x['previous']==head
 for k in ['inputs_shape','responses_shape']:assert x[k]==[2112,32] and all(type(n) is int for n in x[k])
 for k in ['inputs_sha256','responses_sha256']:assert re.fullmatch('[0-9a-f]{64}',x[k])
 assert claimed==hashlib.sha256(canon(x)).hexdigest();head=claimed
assert C['delivery_chain']=={'bytes':len(chainbytes),'head':head,'pairs':204,'sha256':hashlib.sha256(chainbytes).hexdigest()}
assert P['committed_pairs']==204 and type(P['committed_pairs']) is int and P['chain_head']==head
check('ordered204_native_delivery_chain_hashes_shapes_and_progress_prefix')
assert C['status']==A['status']==V['status']=='completed' and C['view']==V
for phase,arrays,metadata in [('validation',408,4),('delivery',408,0),('completion',0,4)]:
 c=V[phase]
 for k,n in [('archives_attempted',408),('archives_read',408),('arrays_validated',arrays),('metadata_attempted',metadata),('metadata_read',metadata),('bytes_read',220703928)]:assert native(c[k],n)
 assert native(c['metadata_bytes'],180415 if metadata else 0)
for k,n in [('pairs',204),('rows',430848),('payload_bytes',220594176)]:assert native(V['delivery'][k],n)
assert set(V['jobs'])=={f'job{i:02d}' for i in range(6)}
for j in V['jobs'].values():assert j['status']=='completed' and native(j['delivered_batches'],34)
assert V['metadata_sha256']==config['public_metadata'] and native(V['allowed_archives'],408) and native(V['raw_payload_bound'],220594176)
assert V['mapping']=={f'job{i:02d}':f'c{i//2:02d}-'+('fp64' if i%2==0 else 'fp32') for i in range(6)}
assert P['view']['delivery']==V['delivery'] and P['view']['validation']==V['validation'] and P['view']['jobs']==V['jobs']
assert P['view']['status']=='delivering' and P['view']['completion']['archives_read']==0
assert C['marker_consumed'] is True and C['uncommitted_delivery'] is False and A['controller_counters_unknown'] is False and A['resource_receipt_unavailable'] is False
for c in [C,V,A]:assert c['certificate'] is False and c['private_gate']=='closed'
for c in [C,V]:assert native(c['new_queries'],0) and native(c['new_fits'],0)
check('native_complete_counters_1224_reads_816_decodes_and_partial_vs_terminal_semantics')
assert A['controller_sha256']==sha(D/'controller-report.json')
for filename,key in [('scope-receipt.json','scope_receipt'),('transport-receipt.json','transport_receipt')]:assert obj(D/filename)==C[key]
tree(C['scope_receipt'],'passed');tree(C['transport_receipt'],'passed');tree(A['receipt'],'completed')
for k in ['free_bytes_before','free_bytes_before_marker','free_bytes_after']:assert type(C[k]) is int and C[k]>=2*1024**3
assert 0<C['elapsed_before_final_write_seconds']<180
check('saved_actual_tree_receipts_disk_limits_and_controller_elapsed_binding')
transport=D/'transport';tm=obj(transport/'manifest.json',4096);assert sha(transport/'manifest.json',4096)==C['transport_receipt']['manifest_sha256']
assert {p.name for p in transport.iterdir()}=={'manifest.json','inputs.npz','responses.npz'}
for rec,field in zip(tm['records'],['inputs','responses'],strict=True):
 assert rec['name']==field+'.npz' and rec['field']==field and rec['shape']==[2112,32]
 assert native(rec['bytes'],(transport/rec['name']).stat().st_size) and 0<rec['bytes']<=544768
 assert sha(transport/rec['name'],544768)==rec['digest']
expectedfiles={'controller-report.json','scope-receipt.json','transport-receipt.json','descriptor-seal.json','progress.json','view-report.json','delivery.jsonl','outer-monitor.json'}
assert {p.name for p in D.iterdir()}==expectedfiles|{'transport','scope-engine','transport-engine'}
for name in ['scope-engine','transport-engine']:assert not list((D/name).iterdir())
for name in expectedfiles:assert (D/name).stat().st_size<=131072
retained=sum((D/n).stat().st_size for n in expectedfiles)+sum(p.stat().st_size for p in transport.iterdir())
assert retained<=config['inventory']['new_artifacts_upper_bound'] and config['inventory']['combined_artifacts_upper_bound']==679665861<1024**3
check('exact_output_layout_caps_literal_transport_hashes_and_retained_postouter_bytes')
copy=R/'experiments/evidence/variable-response-view-v1/actual'
for name in expectedfiles|{'descriptor-seal.json'}:assert sha(copy/name)==sha(D/name)
assert sha(copy/'attempt-marker.json')==sha(markerpath) and sha(copy/'transport/manifest.json')==sha(transport/'manifest.json')
check('durable_small_receipt_copy_matches_actual_terminal_outputs')
result={'approved':True,'saved_engineering_receipt_audit_approved':True,'actual_source_commit':S,'checks':checks,'source_files_verified':21,'runtime_pins_verified':7,'descriptors_verified':408,'committed_pairs':204,'vector_rows':430848,'archive_read_count_reported':1224,'array_decodes_reported':816,'encoded_bytes_per_pass_reported':220703928,'payload_bytes_delivered_reported':220594176,'chain_head':head,'retained_postouter_bytes':retained,'actual_outer_tree_receipt':A['receipt'],'controller_elapsed_before_final_write_seconds':C['elapsed_before_final_write_seconds'],'audit_wall_seconds':time.monotonic()-started,'audit_resources_measured':False,'audit_execution':'one small stdlib saved-receipt checker, no numerical worker or new probe','actual_raw_transcript_reads_by_auditor':0,'new_queries':0,'new_fits':0,'certificate':False,'private_gate':'closed','artifact_sha256':commitments,'limitations':['Saved receipts/source/fixture provenance only. No independent recomputation of array values, raw archive hashes, decoded delivery digests or oracle responses; original 408 transcripts were not reopened.','Marker and source hashes bind the declared ordering; receipt-only audit cannot independently prove filesystem event chronology or every original read.','Actual TREE RSS is sampled, not a continuous maximum or OS reservation. Controller elapsed is before final write; no exact total elapsed inferred.','Only two retained literal transport files were hashed; no original source arrays/context/seed/reference/evaluator/heldout/private files, RNG, model, optimizer, fit or target query were accessed.','This validates an engineering observation pipeline, not model reconstruction, fidelity, gate identity, coverage or private-model readiness.']}
(O/'outcome-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(O/'outcome-audit.md').write_text('אושר אודיט הקבלות ההנדסיות השמורות: אומתו 21 חתימות מקור במקור הריצה ובמאגר הנוכחי, 7 תלותי runtime, התחייבויות ציבוריות, marker ו־descriptor seal של 408 קבצים, שרשרת 204 זוגות ומוני 430848 שורות, 1224 קריאות מדווחות ו־816 decodes. קבלות ה־TREE המקוריות תקינות; outer מדווח 76267520 בתים ב־24 דגימות, וזמן הבקר 1.306990208 שניות לפני כתיבה סופית. פריסת הפלט, גבולותיו, חתימות שני קובצי transport המילוליים והעותקים הקטנים תואמים. לא נפתח אף תמליל מקור ולא חושבו מחדש ערכי מערכים, delivery digests או תגובות oracle; מסקנת האודיט מוגבלת לקבילות המקור/הקבלות/ה־fixtures, ולא מוכיחה את כל אירועי הקריאה או כרונולוגייתם באופן עצמאי. RSS מדוגם אינו שיא רציף. אין ניסוי מספרי, RNG, query או fit ואין טענת שחזור או מוכנות למודל הפרטי; השער נשאר סגור.\n')
files=['audit.py','outcome-audit.json','outcome-audit.md'];(O/'copy-manifest.json').write_text(json.dumps({'files':{n:{'sha256':hashlib.sha256((O/n).read_bytes()).hexdigest(),'bytes':(O/n).stat().st_size} for n in files},'raw_arrays_included':False},indent=2)+'\n')
print(json.dumps({'approved':True,'checks':len(checks),'retained_postouter_bytes':retained,'hashes':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['outcome-audit.json','outcome-audit.md','copy-manifest.json']}}))
