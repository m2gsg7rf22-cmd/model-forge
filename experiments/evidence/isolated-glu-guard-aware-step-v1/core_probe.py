import importlib.util,json
from pathlib import Path
import numpy as np
repo=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
spec=importlib.util.spec_from_file_location('independent_step',repo/'examples/isolated_glu/guard_aware_step.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=np.random.default_rng(771)
g=rng.normal(size=(3,2));left=rng.normal(size=(3,2));right=rng.normal(size=(3,2));c=left[:,:,None]*right[:,None,:]
x=rng.normal(size=(8,2));y=rng.normal(size=(8,2));w=np.repeat([.2,.7,1.3,2.1],2)
# independent logistic forward/complex-step gradient, all other units retained
sigma=lambda t:1/(1+np.exp(-t))
def forward(a):
 t=x@a.T
 return np.einsum('nu,nuo->no',t*sigma(t),np.einsum('nd,uod->nuo',x,c))
res=y*w[:,None]-forward(g)*w[:,None];u=1;t=x@g[u];s=.5*(1+np.tanh(t/2));prime=s+t*s*(1-s);branch=x@c[u].T
d=-2*((w*prime*np.sum(res*branch,axis=1))@x)
fd=[]
for j in range(2):
 a=g.astype(complex);a[u,j]+=1e-25j;e=(forward(a)-y)*w[:,None];fd.append(np.imag(np.sum(e*e))/1e-25)
relative=float(np.linalg.norm(d-fd)/max(1.,np.linalg.norm(fd)))
assert relative<1e-12
# Direct independent componentwise guard-bound boundary comparison
nr=[1.0]*4;na=[2.0]*4;cr=[1.]*4;ca=[2.]*4;ir=[.9]*4;ia=[1.5]*4
assert not m.previous.guard_ok(nr,na,cr,ca,ir,ia)
result={'synthetic_only':True,'queries':0,'actual_fits':0,'independent_complex_step_gradient_relative_error':relative,'initial_guard_not_replaced_by_current':True,'approved_actual_fit':False}
Path('/private/tmp/modelforge-guard-aware-step-review/core-probe-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
