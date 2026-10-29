from pathlib import Path
from unittest.mock import patch
import sys,json,hashlib,math,io
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-joint-audit-math-review');sys.path.insert(0,str(R/'examples/isolated_glu'));import joint_response_audit_math as m;import joint_response_audit_projection as q
checks={}
x=np.array([[.2,-.1],[-.3,.7]]);g=np.array([[.4,-.2],[.3,.6]]);u=np.array([[.8,.1],[-.5,.2]]);d=np.array([[.2,-.7],[.6,.9]])
ref=np.array([[sum((sum(float(row[j])*float(g[k,j]) for j in range(2))/(1+math.exp(-sum(float(row[j])*float(g[k,j]) for j in range(2)))))*sum(float(row[j])*float(u[k,j]) for j in range(2))*float(d[out,k]) for k in range(2)) for out in range(2)] for row in x]);diff=float(np.max(np.abs(m.forward(x,g,u,d)-ref)));assert diff<1e-15
sat=float(m.forward(np.array([[1000.]]),np.ones((1,1)),np.ones((1,1)),np.ones((1,1)))[0,0]);assert abs(sat-1e6)<1e-8
try:m.forward(np.array([[-1000.]]),np.ones((1,1)),np.ones((1,1)),np.ones((1,1)))
except FloatingPointError:pass
else:raise AssertionError('unrepresentable nonzero SiLU accepted')
checks['independent_scalar_forward_and_positive_saturation_fix_negative_underflow_policy']=True
pred=np.array([[5.,-2.],[1.,3.]]);target=np.array([[2.,-1.],[1.,1.]])
a=m.metrics([(pred,target)]);b=m.metrics([(pred[:1],target[:1]),(pred[1:],target[1:])]);assert abs(a['mse']-3.5)<1e-14 and abs(a['half_mse']-1.75)<1e-14 and abs(a['relative_error']-math.sqrt(2))<1e-14
assert all(a[k]==b[k] for k in a) and abs(m.rms([target])-math.sqrt(7/4))<1e-15
assert m.rms([np.array([[1e200,-1e200]])])==1e200
z=np.zeros((1,1));assert m.metrics([(z,z)])['relative_error'] is None and m.metrics([(np.ones((1,1)),np.ones((1,1)))])['relative_error']==0
for f in [lambda:m.metrics([(np.array([[1e-200]]),z)]),lambda:m.metrics([(np.array([[1e200]]),z)]),lambda:m.metrics([(np.array([[1e300],[0.]]),np.array([[1e300],[1e-100]]))])]:
 try:f()
 except FloatingPointError:pass
 else:raise AssertionError('invalid/nonrepresentable nonzero metric accepted')
checks['scaled_streaming_literal_metrics_partition_hugeRMS_zero_and_falsezero_guards']=True
ref1=np.zeros((4,4));ref1[0,0]=1e6;assert not m.array_agrees(ref1+7e-5,ref1)
ref2=np.full((4,4),1e6);val=ref2.copy();val[0,0]+=2e-4;assert not m.array_agrees(val,ref2)
assert m.array_agrees(ref2+1e-6,ref2)
assert m.selection([2.,1.,1.],[2.,1.,1.])=={'index':1,'ambiguous':True,'ambiguous_with':[2]}
assert not m.selection([1.,1.+3e-10],[1.,1.+3e-10])['ambiguous'] and m.selection([1.,1.+2e-10],[1.,1.+2e-10])['ambiguous']
checks['both_max_and_frobenius_tolerances_plus_earliest_and_neartie_rules']=True
# Tiny nine-field archive; moments deliberately contain NaN, outside numeric projection scope.
fields={prefix+n:np.array([[float('nan') if prefix else float(i+1)]]) for prefix in ['','m_','v_'] for i,n in enumerate(['g','u','d'])}
buf=io.BytesIO();np.savez(buf,**fields);calls=[];original=np.frombuffer
def tracked(*a,**kw):calls.append(1);return original(*a,**kw)
with patch.object(np,'frombuffer',tracked):out=q.decode_checkpoint(buf.getvalue(),1,n=1,outputs=1)
assert len(calls)==3 and set(out)=={'g','u','d'} and [float(out[n][0,0]) for n in ['g','u','d']]==[1.,2.,3.] and all(not a.flags.writeable for a in out.values())
fields['v_d']=np.zeros((2,2));bad=io.BytesIO();np.savez(bad,**fields)
with patch.object(np,'frombuffer',side_effect=AssertionError('array before last moment header validated')):
 try:q.decode_checkpoint(bad.getvalue(),1,n=1,outputs=1)
 except ValueError:pass
 else:raise AssertionError('wrong last moment shape accepted')
checks['all9headers_before_views_exact3GUD_views_no_moment_value_claim']=True
result={'passed':True,'checks':checks,'maximum_literal_forward_difference':diff,'positive1000_result':sat,'scope':'small deterministic literal arrays/NPZ bytes only; no original observations/RNG/fit','actual_data_reads':0,'rng_calls':0,'fits':0};(O/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'checks':len(checks),'forward_diff':diff}))
