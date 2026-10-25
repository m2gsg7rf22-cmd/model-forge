from pathlib import Path
import hashlib,json,subprocess
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');B=R.parent/'experiments';D=B/'fp32-hessian-coupling-polish-v1-20261025-0c45745';P=D/'candidate';O=Path('/private/tmp/modelforge-hessian-polish-review');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:dict(np.load(p,allow_pickle=False))
r=json.loads((D/'report.json').read_text());review=json.loads((O/'source-review.json').read_text());b=r['stages']['build'];e=r['stages']['evaluate'];fit=json.loads((P/'fit-report.json').read_text());ev=json.loads((P/'evaluation-report.json').read_text())
assert r['reviewed_sources']==review['sources']
for p,h in review['sources'].items():
 assert sha(R/p)==h and hashlib.sha256(subprocess.check_output(['git','show',r['source_commit']+':'+p],cwd=R)).hexdigest()==h
for stage,inner in [('build',fit),('evaluate',ev)]:
 receipt=json.loads((D/(stage+'-receipt.json')).read_text());assert receipt==r['stages'][stage]
 for k,v in inner.items():assert receipt[k]==v
assert sha(Path('/private/tmp/run_hessian_polish.py'))==review['controller_sha256']
assert r['input_inventory_before']==r['input_inventory_after']
for p,h in r['input_inventory_before'].items():assert sha(B/p)==h
assert len(r['input_inventory_before'])==13
old=B/'fp32-fullmap-dev-v1-20261025-8d2d457/public';original=B/'fp32-cubic-richardson-v1-20261025-188a03a/candidate';paths={'hessians':old/'hessians.npz','calibration_manifest':old/'manifest.json','candidate':original/'candidate.npz','fit_report':original/'fit-report.json'}
assert {k:sha(p) for k,p in paths.items()}==b['inputs_before']==b['inputs_after']
for p,key in [('coupling-proposals.npz','proposals_sha256'),('hessian-residuals.npz','residuals_sha256'),('candidate.npz','candidate_sha256')]:assert sha(P/p)==b[key]
assert sha(P/'public/manifest.json')==b['calibration_manifest_sha256']
assert sha(P/'public-validation.npz')==e['validation_sha256'] and b['candidate_sha256']==e['candidate_sha256']
a=load(paths['candidate']);c=load(P/'candidate.npz');h=load(paths['hessians'])['hessian'];prop=load(P/'coupling-proposals.npz');res=load(P/'hessian-residuals.npz');v=load(P/'public-validation.npz');g=a['gates'];x=np.random.default_rng(20261024).normal(size=(64,32))*4
assert c['gates'].dtype==g.dtype and c['gates'].shape==g.shape and c['gates'].tobytes()==g.tobytes()
def op(i,C):
 t=x@g[i];s=1/(1+np.exp(-t));beta=s+t*s*(1-s);alpha=s*(1-s)*(2+t*(1-2*s));gg=np.outer(g[i],g[i]);value=x@C.T
 return (alpha[:,None]*value)[:,:,None,None]*gg[None,None,:,:]+beta[:,None,None,None]*(np.einsum('i,oj->oij',g[i],C)+np.einsum('oi,j->oij',C,g[i]))[None,:,:,:]
def norm(y):return float(np.sum(y*y))
def same(value,expected):assert np.isclose(value,expected,rtol=1e-10,atol=1e-11),(value,expected)
def rank1(C):
 assert np.isfinite(C).all();s=np.linalg.svd(C,compute_uv=False);assert s[0]>0 and s[1]<1e-10*s[0]
initial=h.copy()
for i in range(64):initial-=op(i,a['couplings'][i])
initerr=float(np.linalg.norm(initial-res['initial'])/np.linalg.norm(initial));assert initerr<1e-10
base=norm(initial);same(base,b['polish']['initial_cost']);scores=[];maxcosterror=0.
assert prop['screen'].shape==(64,32,32) and prop['sequential'].shape==(8,32,32)
for i,row in enumerate(b['polish']['screening']):
 assert row['unit']==i;rank1(prop['screen'][i]);partial=initial+op(i,a['couplings'][i]);cost=norm(partial-op(i,prop['screen'][i]));same(cost,row['initial_proposed_cost']);same(base-cost,row['initial_decrease']);maxcosterror=max(maxcosterror,abs(cost-row['initial_proposed_cost']));scores.append(base-cost)
selected=[i for i in sorted(range(64),key=lambda i:(-scores[i],i)) if scores[i]>max(1e-12*base,1e-24)][:8];assert selected==b['polish']['selected_units']
current=a['couplings'].copy();residual=initial.copy();accepted=[]
for pos,(i,row) in enumerate(zip(selected,b['polish']['decisions'])):
 assert row['unit']==i;rank1(prop['sequential'][pos]);oldcost=norm(residual);partial=residual+op(i,current[i]);newres=partial-op(i,prop['sequential'][pos]);newcost=norm(newres);margin=max(1e-12*oldcost,1e-24);retain=newcost<oldcost-margin
 same(oldcost,row['old_cost']);same(newcost,row['proposed_cost']);same(margin,row['required_decrease']);assert retain==row['accepted']
 if retain:current[i]=prop['sequential'][pos];residual=newres;accepted.append(i)
assert accepted==b['polish']['accepted_units'] and np.array_equal(current,c['couplings'])
final=h.copy()
for i in range(64):rank1(c['couplings'][i]);final-=op(i,c['couplings'][i])
finalerr=float(np.linalg.norm(final-res['final'])/np.linalg.norm(final));assert finalerr<1e-10;same(norm(final),b['polish']['final_cost']);assert b['polish']['local_rankone_solves']==72 and b['polish']['sweeps']==1 and b['direction_updates']==0 and b['new_calibration_calls']==0
rng=np.random.default_rng(941029);expected=np.vstack([rng.normal(size=(512,32))*s for s in [.01,1.,10.,100.] ]);assert np.array_equal(v['inputs'],expected);assert all(z.shape==(2048,32) and np.isfinite(z).all() for z in v.values());assert np.array_equal(v['reference'],v['reference'].astype(np.float32).astype(np.float64))
t=v['inputs']@c['gates'].T;pred=np.einsum('nw,wod,nd->no',t*.5*(1+np.tanh(t*.5)),c['couplings'],v['inputs'],optimize=True);forwarderr=float(np.linalg.norm(pred-v['prediction'])/np.linalg.norm(v['prediction']));assert forwarderr<1e-12
groups=[]
for k,row in enumerate(e['groups']):
 sl=slice(512*k,512*(k+1));delta=v['prediction'][sl]-v['reference'][sl];rel=float(np.linalg.norm(delta)/np.linalg.norm(v['reference'][sl]));absolute=float(abs(delta).max());same(rel,row['relative_frobenius']);same(absolute,row['max_absolute']);assert rel>.001;groups.append({'scale':row['scale'],'relative_frobenius':rel,'max_absolute':absolute})
assert e['vector_calls']==2048 and not e['all_groups_relative_at_most_1e_minus3'] and not e['sealed'] and e['private_gate']=='closed'
meta=json.loads(paths['calibration_manifest'].read_text());prior_fit=json.loads(paths['fit_report'].read_text());assert meta['files']['hessians.npz']==b['inputs_before']['hessians'] and meta['channel']=='fp32_compute'
assert prior_fit['candidate_built'] and prior_fit['candidate_sha256']==b['inputs_before']['candidate'] and prior_fit['calibration_manifest_sha256']==b['inputs_before']['calibration_manifest']
assert prior_fit['method']=='isolated-glu-fp32-cubic-richardson-v1' and b['method']=='isolated-glu-fp32-hessian-coupling-polish-v1'
marker=json.loads((P/'polish-attempt.json').read_text());assert marker=={'automatic_retry':False,'max_local_solves':72,'new_calibration_calls':0}
eval_marker=json.loads((P/'evaluation-attempt.json').read_text());assert eval_marker['public_reference_calls_reserved']==2048 and eval_marker['automatic_retry'] is False
result={'approved':True,'scope':'independent saved-array outcome audit; quality FAIL, no promotion','report_sha256':sha(D/'report.json'),'source_commit':r['source_commit'],'source_and_dependency_count':len(review['sources']),'old_inventory_count_unchanged':13,'primary_inputs_verified':b['inputs_before'],'candidate_sha256':b['candidate_sha256'],'proposal_sha256':b['proposals_sha256'],'all64_proposal_costs_verified':True,'max_screening_cost_absolute_discrepancy':maxcosterror,'selected_and_accepted':selected,'all8_retention_decisions_verified':True,'local_solves_reported':72,'initial_cost':base,'final_cost':norm(final),'relative_cost_reduction':1-norm(final)/base,'fresh_initial_residual_relative_discrepancy':initerr,'fresh_final_residual_relative_discrepancy':finalerr,'gates_byte_identical':True,'all_couplings_rankone_nonzero_finite':True,'candidate_forward_relative_discrepancy':forwarderr,'evaluation_seed':941029,'evaluation_points':2048,'quality':'FAILED','groups':groups,'new_oracle_queries_by_reviewer':0,'new_coupling_solves_by_reviewer':0,'old_evaluation_arrays_read':False,'resources':{'seconds':r['controller_seconds'],'build_sampled_rss_bytes':b['peak_sampled_rss_bytes'],'build_samples':b['memory_samples'],'evaluation_sampled_rss_bytes':e['peak_sampled_rss_bytes'],'evaluation_samples':e['memory_samples'],'actual_peak_known':False},'limitations':['Saved-Hessian objective reduction does not establish forward generalization.','Different public validation points prevent causal comparison with the prior candidate on prior points.','Seed941029 now consumed; do not reuse as independent validation.','Polled RSS cannot prove continuous hard memory bound.'],'private_gate':'closed'}
(O/'outcome-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('review_sha256',sha(O/'outcome-review.json'))
