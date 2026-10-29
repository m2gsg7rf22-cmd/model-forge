from pathlib import Path
import sys,json,hashlib,math
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-joint-response-kernel-review');sys.path.insert(0,str(R/'examples/isolated_glu'))
import joint_response_kernel as k
checks={}
# Independent scalar, complex-analytic loss: no product forward/gradient methods.
def scalar(a,x):
 return np.array([[sum((sum(row[j]*a['g'][t,j] for j in range(len(row)))/(1+np.exp(-sum(row[j]*a['g'][t,j] for j in range(len(row))))))*sum(row[j]*a['u'][t,j] for j in range(len(row)))*a['d'][out,t] for t in range(len(a['g']))) for out in range(len(a['d']))] for row in x])
def refgrad(a,x,y):
 grad={}
 for name in a:
  v=np.zeros_like(a[name])
  for i in np.ndindex(v.shape):
   b={n:val.astype(complex) for n,val in a.items()};b[name][i]+=1e-30j
   e=scalar(b,x)-y;v[i]=np.imag(.5*np.mean(e*e))/1e-30
  grad[name]=v
 return grad
A={'g':np.array([[.2,-.3],[.5,.1]]),'u':np.array([[.7,.4],[-.2,.8]]),'d':np.array([[.6,-.1],[-.4,.9]])}
x=np.array([[.3,-.2],[.7,.4],[-.1,.5]]);y=np.array([[.4,-.2],[.1,.7],[-.3,.6]])
p=k.Parameters(**A);loss,grad=k.loss_gradient(p,x,y);reference=refgrad(A,x,y)
forward_diff=float(np.max(np.abs(k.predict(p,x)-scalar(A,x))));gradient_diff=max(float(np.max(np.abs(getattr(grad,n)-reference[n]))) for n in A)
assert forward_diff<1e-15 and gradient_diff<1e-14
checks['scalar_forward_and_all_coordinate_complex_step_gradients']=True
# Two actual literal-prefix Adam updates versus independent complex-step gradients.
schedule=np.zeros((4000,256),dtype=np.int64);schedule[1,:]=1;events=[]
class Stop(Exception):pass
def observe(e):
 events.append(e)
 if isinstance(e,k.Step) and e.step==2:raise Stop
try:k.fit(p,x,y,x,y,schedule,observe)
except Stop:pass
else:raise AssertionError('callback exception swallowed')
state={n:a.copy() for n,a in A.items()};m={n:np.zeros_like(a) for n,a in A.items()};v={n:np.zeros_like(a) for n,a in A.items()};adamdiff=0.
steps=[e for e in events if isinstance(e,k.Step)]
for t,e in enumerate(steps):
 g=refgrad(state,x[t:t+1],y[t:t+1]);lr=.0001+.5*(.01-.0001)*(1+math.cos(math.pi*t/3999))
 for n in state:
  m[n]=.9*m[n]+.1*g[n];v[n]=.999*v[n]+.001*g[n]*g[n]
  state[n]=state[n]-lr*(m[n]/(1-.9**(t+1)))/(np.sqrt(v[n]/(1-.999**(t+1)))+1e-8)
  adamdiff=max(adamdiff,float(np.max(np.abs(state[n]-getattr(e.parameters,n)))))
assert len(steps)==2 and adamdiff<1e-14 and len([0,*range(256,4000,256),4000])==17
assert k.learning_rate(0)==.01 and abs(k.learning_rate(3999)-.0001)<1e-18
checks['two_independent_adam_steps_schedule_endpoints_and_callback_propagation']=True
sx,sy=k.training_scales(x,y);physical=k.physical_parameters(p,sx,sy);fold_diff=float(np.max(np.abs(k.predict(physical,x)-sy*scalar(A,x/sx))))
assert fold_diff<1e-14
for a in [p.g,p.u,p.d]:
 try:a.setflags(write=True)
 except ValueError:pass
 else:raise AssertionError('mutable parameters')
checks['physical_coordinate_folding_and_immutable_parameters']=True
# Exact zero target has undefined relative metric; only genuine zero residual has zero MSE.
z=np.zeros((1,1));zero=k.Parameters(z,z,z);assert k._metric(zero,np.zeros((1,1)),z)==(0.,None)
small=np.zeros((8,1));small[0,0]=float.fromhex('0x0.0000000000001p-1022')
huge=k.Parameters(np.ones((1,1)),np.ones((1,1)),np.array([[1e308]]));hx=np.array([[1.],[0.]]);hy=np.array([[1e308/(1+np.exp(-1.))],[1e-100]])
for fn in [lambda:k._metric(zero,np.zeros((8,1)),small),lambda:k.loss_gradient(zero,z,np.array([[1e-200]])),lambda:k._metric(huge,hx,hy)]:
 try:fn()
 except FloatingPointError:pass
 else:raise AssertionError('nonzero metric silently underflowed to zero')
checks['all_three_previously_observed_false_zero_boundaries_now_fail_closed']=True
try:k.fit(zero,np.zeros((8,1)),small,np.zeros((8,1)),small,np.zeros((4000,256),dtype=np.int64),lambda e:(_ for _ in ()).throw(AssertionError('observer before numerical failure')))
except k.NumericalFailure as e:assert e.updates_completed==0
else:raise AssertionError('numerical failure became prefix success')
checks['initial_metric_numeric_failure_has_no_observer_or_success_prefix']=True
result={'passed':True,'checks':checks,'maximum_differences':{'forward':forward_diff,'gradient':gradient_diff,'two_step_adam':adamdiff,'physical_folding':fold_diff},'scope':'new tiny literal data only; no registered observation/RNG/teacher/query; two-update synthetic prefix only','source_sha256':hashlib.sha256((R/'examples/isolated_glu/joint_response_kernel.py').read_bytes()).hexdigest()}
(O/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
