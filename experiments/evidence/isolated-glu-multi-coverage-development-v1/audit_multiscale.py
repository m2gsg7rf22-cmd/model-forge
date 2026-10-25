from pathlib import Path
import hashlib,json,time
import numpy as np
repo=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository'); run=Path('/Users/adamfaruhi123/.local/share/modelforge-research/experiments/isolated-glu-multi-coverage-development-v1-20261025'); old=Path('/Users/adamfaruhi123/.local/share/modelforge-research/experiments/isolated-glu-fullmap-v3-20261025-a0d0556/public'); out=Path('/private/tmp/modelforge-multi-independent-review')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def rel(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def student(g,c,x):
 y=np.zeros((len(x),32))
 for a,b in zip(g,c):
  z=x@a;y+=(z*.5*(1+np.tanh(z/2)))[:,None]*(x@b.T)
 return y
plan=read(run/'multiscale-plan.json');assert sha(repo/'examples/isolated_glu/multiscale.py')==plan['source_sha256'];rows=[];start=time.perf_counter()
for k in range(3):
 p=run/str(k)/'calibration-fullmap'; report=read(p/'multiscale-solve.json');monitor=read(run/f'monitored-multiscale-{k}.json');assert all(monitor[n]==v for n,v in report.items());assert monitor['index']==k;assert monitor['input_direction_sha256']==sha(run/str(k)/'multi-coverage-solution.npz');assert monitor['observation_sha256']==sha(old/str(k)/'multiscale-observations.npz')==sha(p/'multiscale-observations.npz')
 assert monitor['process_monitor_scope']=='process_and_children' and monitor['memory_samples']>0 and monitor['memory_limit_mb']==512 and 0<monitor['peak_sampled_rss_bytes']<=512*1024**2 and 0<=monitor['seconds']<180
 v=load(p/'refined-solution.npz')['directions'];assert np.array_equal(v,load(run/str(k)/'multi-coverage-solution.npz')['directions']);data=load(p/'multiscale-observations.npz');sol=load(p/'multiscale-solution.npz');assert set(sol)=={'gates','couplings'};g,c=sol['gates'],sol['couplings'];assert g.shape==(64,32) and c.shape==(64,32,32);assert np.isfinite(g).all() and np.isfinite(c).all()
 magn=np.linalg.norm(g,axis=1);assert (magn>0).all();assert np.allclose(np.abs(np.sum(g*v,axis=1))/magn,1,atol=1e-10,rtol=0);sv=np.linalg.svd(c,compute_uv=False);tail=float(np.max(sv[:,1]/sv[:,0]));assert tail<1e-10;assert np.isclose(magn.min(),report['magnitude_min']) and np.isclose(magn.max(),report['magnitude_max'])
 # Rank audit of the small polynomial design, without solving/fitting its coefficients.
 x=data['small_x'];design=(((x@v.T)**2)[:,:,None]*x[:,None,:]).reshape(4096,2048);s=np.linalg.svd(design,compute_uv=False);rank=int((s>s[0]*max(design.shape)*np.finfo(float).eps).sum());cond=float(s[0]/s[-1]);assert rank==report['small_design_rank']==report['small_design_columns']==2048;assert np.isclose(cond,report['small_design_condition'],rtol=1e-7);del design
 # Independent even-quadratic reconstruction from only the saved canonical outputs.
 even=.5*(data['even_pos']+data['even_neg']);q=np.zeros((32,32,32));q[:,np.arange(32),np.arange(32)]=even[:32].T;row=32
 for i in range(32):
  for j in range(i+1,32):q[:,i,j]=q[:,j,i]=.5*(even[row]-even[i]-even[j]);row+=1
 ii,jj=np.triu_indices(32);unsigned=v*magn[:,None];a=.25*(np.einsum('kod,ke->kode',c,unsigned)+np.einsum('koe,kd->kode',c,unsigned));oridesign=a[:,:,ii,jj].transpose(1,2,0).reshape(-1,64);sor=np.linalg.svd(oridesign,compute_uv=False);orrank=int((sor>sor[0]*max(oridesign.shape)*np.finfo(float).eps).sum());assert orrank==report['orientation_rank']==64
 signs=np.sign(np.sum(g*v,axis=1));ores=rel(oridesign@signs,q[:,ii,jj].reshape(-1));assert np.isclose(ores,report['orientation_rounded_relative_residual'],atol=1e-12,rtol=1e-7)
 metrics={}
 for name,scale in [('small',.001),('large',1000.)]:
  x=data[name+'_x']*scale;odd=.5*(student(g,c,x)-student(g,c,-x));metrics[name+'_saved_odd_relative_frobenius_error']=rel(odd,data[name+'_odd'])
 assert report['magnitude_design_rank']==64 and report['magnitude_min']>0 and report['public_only_fit'] is True
 rows.append({'index':k,'source_observation_direction_commitments_match':True,'prefix_and_directions_identical':True,'solution_sha256':sha(p/'multiscale-solution.npz'),'student_arrays_finite':True,'student_rankone_max_relative_tail':tail,'small_design_independent_rank':rank,'small_design_independent_condition':cond,'orientation_design_independent_rank':orrank,'magnitude_design_reported_rank':64,'magnitude_design_exact_reaudit_not_available':'UnprojectedBcoefficients are not saved; no refit performed.','orientation_rounded_independent_residual':ores,'calibration_only_student_metrics':metrics,'seconds':monitor['seconds'],'peak_sampled_rss_bytes':monitor['peak_sampled_rss_bytes']})
result={'status':'verified_public_calibration_integration','source_sha256':plan['source_sha256'],'plan_sha256':sha(run/'multiscale-plan.json'),'results':rows,'audit_seconds':time.perf_counter()-start,'no_fitting_or_new_queries':True,'no_private_teacher_or_heldout_read':True,'teacher_identity_or_fullmap_validated':False,'formal_v3_outcome_changed':False,'private_final_model_gate':'closed','blocking_findings':[],'limits':['Allmetrics are on saved calibration inputs and thus cannot establish heldout generalization.','Rankonecouplings are a projected structural constraint, not teacherparameter recovery evidence.','Magnitude-design rank64 is a recorded solver diagnostic; original unprojectedB coefficients not saved, so exact independent recomputation would require forbidden refit.','Freshprotocolv4 not yet independently evaluated.']}
(out/'multiscale-addendum.json').write_text(json.dumps(result,indent=2)+'\n');(out/'multiscale-addendum.md').write_text('# תוספת ביקורת — אינטגרציית כיול ציבורית\n\nבשלושת המערכים אומתו המקור, התצפיות,64הכיוונים והמשאבים. מערכי התלמיד סופיים, השערים קולינריים לכיוונים והצימודים מדרגה1. דרגות תכןsmall2048ו־orientation64חושבו עצמאית ללאfitting. שגיאותoddושאריתevenחושבו על הכיול השמור בלבד.\n\nדרגתmagnitude64היא דיווח solver;Bלפניהיטל לא נשמר ולכן לא בוצע חישוב חוזר שלה. אין כאן אימות זהות מורה או דיוקheldout;v3נשאר נכשל והשער הפרטי סגור. פירוט ב־multiscale-addendum.json.\n');print(json.dumps(result,indent=2))
