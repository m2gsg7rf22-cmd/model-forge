import json, time
import numpy as np
start=time.monotonic()
rng=np.random.default_rng(20261024)
d,h=32,64
norm=lambda x: x/np.linalg.norm(x)
g=np.array([norm(x) for x in rng.normal(size=(h,d))])
u=rng.normal(size=(h,d)); u-=np.sum(u*g,axis=1)[:,None]*g;u=np.array([norm(x) for x in u])
vec=lambda x:x.reshape(-1)
blocks=[np.stack([vec(np.outer(a,a)),vec(np.outer(a,b)+np.outer(b,a))],axis=1) for a,b in zip(g,u)]
B=np.concatenate(blocks,axis=1)
Q=np.linalg.qr(B)[0]

def tangent(v):
    L=np.stack([vec(np.outer(v,e)+np.outer(e,v)) for e in np.eye(d)],axis=1)
    return np.linalg.qr(L)[0]

def modes(v):
    T=tangent(v); R=T-Q@(Q.T@T)
    _,s,W=np.linalg.svd(R,full_matrices=False)
    return s,T@W[-2:].T

out={'seed':20261024,'d':d,'h':h,'basis_condition':float(np.linalg.cond(B)),'noise_checks':[]}
s,_=modes(g[0]);out['exact_tangent_smallest']=s[-5:].tolist()
w=norm(rng.normal(size=d)-np.dot(rng.normal(size=d),g[0])*g[0]);w=norm(w-np.dot(w,g[0])*g[0])
for eps in [1e-1,1e-2,1e-3,1e-4]:
    v=norm(g[0]+eps*w);s,_=modes(v)
    r=vec(np.outer(v,v)); r-=Q@(Q.T@r)
    out['noise_checks'].append({'eps':eps,'tangent_two':s[-2:].tolist(),'rank1_objective':float(r@r)})
for eps in [0,1e-3,1e-2]:
    K=[]
    for i in range(h-1):
        v=norm(g[i]+eps*rng.normal(size=d)/np.sqrt(d))
        _,M=modes(v)
        K.append(Q@(Q.T@M))
    K=np.concatenate(K,axis=1);Z=np.linalg.qr(K)[0]
    R=Q-Z@(Z.T@Q)
    A,rs,_=np.linalg.svd(R,full_matrices=False)
    GG=vec(np.outer(g[-1],g[-1]));Gres=GG-Z@(Z.T@GG)
    eig=np.linalg.eigvalsh(Gres.reshape(d,d))
    val,V=np.linalg.eigh((A[:,0].reshape(d,d)+A[:,0].reshape(d,d).T)/2)
    principal=V[:,np.argmax(np.abs(val))]
    out.setdefault('deflation',[]).append({'eps':eps,'residual_singular_first4':rs[:4].tolist(),'residual_frobenius':float(np.linalg.norm(R)),'missing_projected_rank1_tail':float(np.linalg.norm(eig[np.argsort(np.abs(eig))[:-1]])/np.linalg.norm(eig)),'one_basis_top_eigen_cosine_missing':float(abs(principal@g[-1]))})
# Deliberate non-generic counterexample: complete symmetric space.
ds=3;Vs=np.eye(ds*ds) # T is contained, so intersection dimension d, not2.
out['full_space_counterexample']={'d':ds,'intersection_dimension':ds}
out['elapsed_seconds']=time.monotonic()-start
print(json.dumps(out,indent=2))
