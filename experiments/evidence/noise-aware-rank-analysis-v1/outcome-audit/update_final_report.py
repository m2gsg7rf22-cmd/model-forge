from pathlib import Path
import hashlib,json
P=Path('/private/tmp/modelforge-rank-analysis-outcome-review')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
receipt=read(P/'audit-resource-receipt.json');pre=read(P/'tree-scope-preflight-receipt.json');result=read(P/'probe-results.json');old=read(P/'attempt03-results.json')
assert receipt['status']=='passed' and receipt['process_monitor_scope']=='process_and_children'
assert receipt['memory_samples']==26 and receipt['peak_sampled_rss_bytes']==219709440<512*1024**2
assert receipt['audit_launcher_elapsed_seconds']==1.7166753339988645<180
assert pre['process_monitor_scope']=='process_and_children' and pre['memory_samples']>=1
assert sha(P/'probe-results.json')==receipt['result_sha256']=='313e2115548da552b809b53bc1b99281eee4974329ba7524372b54e2f3e1465b'
assert result==old and sha(P/'probe.py')=='02a86c84540182b5fbf44aaa9a531c5182cb424c4b5b28c044133c294c2adbf3'
for src,dst in [('probe.py','attempt04-probe.py'),('audit-resource-receipt.json','attempt04-receipt.json'),('probe-results.json','attempt04-results.json')]:
 (P/dst).write_bytes((P/src).read_bytes())
v=read(P/'outcome-audit.json')
v['audit_attempt_history'][2]['status']='numeric_pass_resource_scope_incomplete'
v['audit_attempt_history'][2]['reason']='All16 numerical cases passed, but children enumeration was unavailable; does not certify TREE resource scope.'
v['audit_attempt_history'].append({'attempt':4,'status':'passed_all16_numeric_cases_and_polled_tree_resources','executed_by':'parent after separate successful synthetic TREE preflight; reviewer did not launch again','change_from_attempt03':'Only two review artifact paths changed to byte-identical durable repository evidence; no mathematical changes','probe_sha256':sha(P/'probe.py'),'receipt':receipt,'result_identical_to_attempt03':True,'result_sha256':sha(P/'probe-results.json')})
assert len(v['audit_attempt_history'])==4
v['audit_attempt_history'][0]['receipt']=read(P/'attempt01-receipt.json')
v['full_audit_resource_protocol_approved']=True
v['audit_tree_memory_bound_verified']=True
v['audit_resources']=receipt
v['audit_resource_preflight']=pre
v['resource_remediation_status']='completed in attempt04 after successful synthetic TREE preflight; original analysis not rerun and original marker unchanged'
v['audit_resource_limitations']='Attempt04 successfully used polled process-and-children RSS with512MiB/180s bounds. Sampled peak219709440 bytes/26samples and launcher elapsed1.7166753339988645s are not a continuous/hard peak or hard real-time guarantee. Prior incomplete attempts remain preserved; no new experiment/forward/query/fitting occurred.'
v['final_probe_sha256']=sha(P/'probe.py')
md=(P/'outcome-audit.md').read_text()
oldpara='משאבי האודיט: הניסיון השלישי השלים16 מקרים בכ־1.677 שניות לפי launcher, process RSS מדוגם219,758,592 bytes ב־28 דגימות. children enumeration לא היה זמין, ולכן TREE bound של האודיט אינו מאומת. לא בוצעה ריצה חוזרת לתיקון scope.'
newpara='משאבי האודיט: הניסיון השלישי השלים16 מקרים בכ־1.677 שניות לפי launcher, עם process RSS מדוגם219,758,592 bytes ב־28 דגימות; children enumeration לא היה זמין ולכן אינו אישור TREE. לאחר preflight סינתטי נפרד עם TREE הפעיל ההורה ניסיון אודיט רביעי בהרשאה מורחבת. ניסיון4 עבר עם process-and-children RSS מדוגם219,709,440 bytes ב־26 דגימות וזמן launcher של1.716675334 שניות, במסגרת512MiB/180s. תוצאתו זהה בבתים לתוצאת ניסיון3: SHA256 313e2115548da552b809b53bc1b99281eee4974329ba7524372b54e2f3e1465b. ניטור מדוגם אינו הוכחת שיא רציף או זמן אמת קשיח.'
assert oldpara in md;md=md.replace(oldpara,newpara)
oldpara2='שני ניסיונות אודיט מוקדמים נשמרו: בראשון חריגה לא נתפסה לפני פלט JSON ו־stderr הזמני אבד; בשני נחשפה שגיאה שלי בהגדרת bytes_reserved, הכוללת גם manifest. שניהם נעצרו לפני מקרה נומרי. תוקן סקריפט האודיט בלבד; הניתוח המקורי וה־marker לא הורצו או שונו. קבלות הכשל וההצלחה חתומות ב־JSON.'
newpara2='כל ארבעת ניסיונות האודיט נשמרו במפורש: (1) חריגה לפני JSON, stderr זמני אבד ומשאבים אינם ידועים; (2) שגיאה שלי בספירת bytes_reserved, הכוללת manifest, נעצר לפני מקרה נומרי; (3) כל החישובים עברו אך scope המשאבים היה חלקי; (4) כל החישובים והניטור TREE המדוגם עברו. בין3 ל־4 השתנו רק שני נתיבי ראיות הביקורת לעותקים זהים בבתים במאגר, ללא שינוי מתמטי. לא הופעל שוב הניתוח המקורי, לא נוצרו תצפיות חדשות וה־marker לא שונה. הבודק לא הפעיל פרוב נוסף לאחר בקשת העצירה; ההורה ביצע את ניסיון4. קבלות הכשל, ההצלחה ו־preflight חתומות ב־JSON.'
assert oldpara2 in md;md=md.replace(oldpara2,newpara2)
(P/'outcome-audit.md').write_text(md)
files=['probe.py','probe-results.json','launcher.py','attempt01-receipt.json','attempt02-receipt.json','attempt03-probe.py','attempt03-receipt.json','attempt03-results.json','attempt04-probe.py','attempt04-receipt.json','attempt04-results.json','audit-resource-receipt.json','tree-scope-preflight-receipt.json','outcome-audit.md']
v['evidence_sha256']={n:sha(P/n) for n in files}
(P/'outcome-audit.json').write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
files+=['outcome-audit.json','update_final_report.py']
manifest={'copy_files_sha256':{n:sha(P/n) for n in files},'exclude':['engine/','tree-scope-engine/','finalize.py (superseded old snapshot generator)','audit-progress.json (temporary progress)','audit-resource-receipt-initial-failed.json (identical to retained attempt01-receipt.json)'],'scope':'Small audit evidence only; do not copy large original arrays','original_analysis_attempts':1,'audit_attempts':4}
(P/'copy-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'approved':True,'polled_tree_resource_protocol_approved':True,'audit_attempts':4,'original_analysis_attempts':1,'outcome_sha256':sha(P/'outcome-audit.json'),'md_sha256':sha(P/'outcome-audit.md'),'copy_manifest_sha256':sha(P/'copy-manifest.json'),'copy_files':len(files)}))
