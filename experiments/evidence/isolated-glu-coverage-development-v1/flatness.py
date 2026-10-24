import numpy as np,json
rng=np.random.default_rng(20261024);d,h=32,64
normalize=lambda x:x/np.linalg.norm(x)
g=np.array([normalize(x) for x in rng.normal(size=(h,d))]);u=rng.normal(size=(h,d));u-=np.sum(u*g,axis=1)[:,None]*g;u=np.array([normalize(x) for x in u])
B=np.concatenate([np.stack([np.outer(a,a).ravel(),(np.outer(a,b)+np.outer(b,a)).ravel()],axis=1) for a,b in zip(g,u)],axis=1);Q=np.linalg.qr(B)[0]
rows=[]
for eps in [1e-1,1e-2,1e-3,1e-4]:
 v=normalize(g[0]+eps*u[0]);vv=np.outer(v,v).ravel();r=vv-Q@(Q.T@vv);T=np.linalg.qr(np.stack([(np.outer(v,e)+np.outer(e,v)).ravel() for e in np.eye(d)],axis=1))[0];s=np.linalg.svd(T-Q@(Q.T@T),compute_uv=False)
 rows.append({'eps':eps,'rank1_objective':float(r@r),'objective_over_eps4':float(r@r/eps**4),'tangent_two':s[-2:].tolist()})
print(json.dumps({'perturbation':'gate toward own value direction','results':rows},indent=2))
