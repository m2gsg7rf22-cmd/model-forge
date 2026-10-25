"""Synthetic four-round audit fixture; no fitting, teacher or heldout arrays."""
from pathlib import Path
import json
import numpy as np

def make_case(m, p: Path, increments=(1, 1, 1, 1)):
    p.mkdir(exist_ok=False);(p/'rescue').mkdir();rng=np.random.default_rng(81273);g=rng.normal(size=(64,32));g/=np.linalg.norm(g,axis=1,keepdims=True);u=rng.normal(size=(64,32));i,j=np.triu_indices(32);w=np.where(i==j,1.,np.sqrt(2.))
    def pack(a):return a[...,i,j]*w
    components=np.stack([pack(a) for x,y in zip(g,u) for a in [np.outer(x,x),np.outer(x,y)+np.outer(y,x)]]);basis=np.linalg.qr(components.T)[0].T;vec=np.zeros((2048,528));vec[:128]=basis;mat=np.zeros((2048,32,32));mat[:,i,j]=vec/w;mat[:,j,i]=mat[:,i,j];h=mat.reshape(64,32,32,32);np.savez(p/'hessians.npz',hessian=h);_,_,vt=np.linalg.svd(pack(h).reshape(-1,528),full_matrices=False);basis=vt[:128]
    def losses(d):
     q=d[:,i]*d[:,j]*w;r=q-(q@basis.T)@basis;return np.sum(r*r,axis=1)
    base={'directions':g[:60].copy(),'all_directions':np.tile(g[:60],(5,1))[:256]};base['residual']=losses(base['all_directions']);np.savez(p/'base-refined-solution.npz',**base);m.write(p/'base-refine.json',{'directions_retained':60});report={'branch':'multi','input_count':60,'base_solution_sha256':m.digest(p/'base-refined-solution.npz'),'base_refine_sha256':m.digest(p/'base-refine.json'),'heldout_read':False,'new_queries':0,'rescue_monitor':{'process_monitor_scope':'process_and_children','memory_samples':1,'peak_sampled_rss_bytes':1000,'memory_limit_mb':512}};m.write(p/'direction-selection.json',report)
    def merge_known(known,candidates,losses):
        selected=list(known)
        for k in np.argsort(losses):
            if losses[k]<=1e-8 and all(abs(candidates[k]@v)<.9995 for v in selected):
                selected.append(candidates[k])
        return np.stack(selected)
    known=base['directions'];rs=[];arrays=[base['all_directions']];ll=[base['residual']];srng=np.random.default_rng(20261025)
    for n, extra in enumerate(increments):
     starts=m.replay_starts(known,basis,srng);cand=np.tile(known,(512//len(known)+1,1))[:512].copy();cand[-extra:]=g[len(known):len(known)+extra];l=losses(cand);after=merge_known(known,cand,l);f=p/f'rescue/round-{n}.npz';np.savez(f,input_directions=known,starts=starts,all_directions=cand,residual=l,directions=after);rs.append({'round':n,'input_count':len(known),'retained_count':len(after),'starts':512,'artifact_sha256':m.digest(f),'geometry':{'component_rank':2*len(known),'residual_dimension':128-2*len(known)}});known=after;arrays.append(cand);ll.append(l)
    np.savez(p/'rescue/multi-coverage-solution.npz',directions=known);np.savez(p/'refined-solution.npz',directions=known,all_directions=np.concatenate(arrays),residual=np.concatenate(ll));rr={'source_sha256':m.registration()['solver_source_sha256']['examples/isolated_glu/coverage_multi.py'],'input_sha256':{'hessians.npz':m.digest(p/'hessians.npz'),'refined-solution.npz':report['base_solution_sha256']},'input_count':60,'retained_count':64,'heldout_read':False,'new_queries':0,'configuration':{'public_seed':20261025,'max_rounds':4,'mixtures_per_round':256,'eigenvectors_per_mixture':2,'gn_steps':24},'rounds':rs};m.write(p/'rescue/report.json',rr)
    def refresh():
     for n,row in enumerate(rr['rounds']):row['artifact_sha256']=m.digest(p/f'rescue/round-{n}.npz')
     (p/'rescue/report.json').write_text(json.dumps(rr));(p/'refine.json').write_text(json.dumps({'base_report_sha256':report['base_refine_sha256'],'rescue_report_sha256':m.digest(p/'rescue/report.json'),'directions_retained':64}))
    refresh()
    return rr, refresh
