from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-gate-gradient-review');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location('gradient',R/'examples/isolated_glu/gate_gradient.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rng=np.random.default_rng(664193);rows=[]
def independent(G,C,X):
 t=X@G.T
 return np.einsum('nw,wod,nd->no',t/(1+np.exp(-t)),C,X,optimize=True)
for dim in (1,3,5):
 G=rng.normal(size=(3,dim))*.5;C=rng.normal(size=(3,2,dim))*.5;X=rng.normal(size=(17,dim));Y=rng.normal(size=(17,2));w=np.exp(rng.uniform(-1,1,size=17));saved=[z.copy() for z in [G,C,X,Y,w]];L,dg,dc=m.loss_gradient(G,C,X,Y,w)
 def loss(g,c):return np.sum(((independent(g,c,X)-Y)*w[:,None])**2)
 assert np.allclose(m.forward(G,C,X),independent(G,C,X),rtol=1e-13,atol=1e-13) and np.isclose(L,loss(G,C),rtol=1e-13)
 # Complex-step independent analytic differentiation avoids centered-difference cancellation.
 gradg=np.empty_like(G);gradc=np.empty_like(C);step=1e-24
 for idx in np.ndindex(G.shape):
  z=G.astype(complex);z[idx]+=step*1j;gradg[idx]=np.imag(loss(z,C))/step
 for idx in np.ndindex(C.shape):
  z=C.astype(complex);z[idx]+=step*1j;gradc[idx]=np.imag(loss(G,z))/step
 ge=float(np.linalg.norm(dg-gradg)/np.linalg.norm(gradg));ce=float(np.linalg.norm(dc-gradc)/np.linalg.norm(gradc));assert ge<1e-11 and ce<1e-11
 E=w[:,None]*(independent(G,C,X)-Y);maxgram=0.;maxjvp=0.
 for i in range(3):
  J=np.column_stack([np.imag(w[:,None]*independent((G[i:i+1].astype(complex)+1j*step*np.eye(dim)[j:j+1]),C[i:i+1],X)).ravel()/step for j in range(dim)])
  gram=m.gate_gram(G[i],C[i],X,w);assert np.allclose(gram,J.T@J,rtol=1e-12,atol=1e-13)
  assert np.allclose(2*J.T@E.ravel(),dg[i],rtol=1e-12,atol=1e-12)
  direction=rng.normal(size=dim);jvp=m.gate_jvp(G[i],C[i],X,w,direction);assert np.allclose(jvp.ravel(),J@direction,rtol=1e-12,atol=1e-13)
  maxgram=max(maxgram,float(np.linalg.norm(gram-J.T@J)/max(np.linalg.norm(gram),1e-30)));maxjvp=max(maxjvp,float(np.linalg.norm(jvp.ravel()-J@direction)/max(np.linalg.norm(jvp),1e-30)))
 L2,g2,c2=m.loss_gradient(G,C,X,Y,2*w);assert np.isclose(L2,4*L) and np.allclose(g2,4*dg) and np.allclose(c2,4*dc)
 for before,after in zip(saved,[G,C,X,Y,w]):assert np.array_equal(before,after)
 zero=m.gate_gram(G[0],np.zeros_like(C[0]),X,w);assert np.array_equal(zero,np.zeros((dim,dim)))
 rows.append({'dimension':dim,'gate_gradient_relative_error':ge,'coupling_gradient_relative_error':ce,'gram_relative_error':maxgram,'jvp_relative_error':maxjvp})
result={'method':'independent complex-step sigmoid/einsum, no target oracle','cases':rows,'all_gate_and_coupling_coordinates_verified':True,'twice_JstarE_equals_gate_gradient':True,'Gram_equals_explicit_JstarJ':True,'weight_doubling_scales_loss_and_both_gradients_by_four':True,'inputs_unmodified':True,'zero_coupling_singular_gram_is_allowed':True,'actual_experiment_arrays_read':0,'teacher_queries':0,'saved_candidate_fits':0,'candidate_updates':0}
(O/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');files=['examples/isolated_glu/gate_gradient.py','tests/test_glu_gate_gradient.py','docs/glu-gate-gradient-primitive.md'];review={'approved':True,'scope':'synthetic-only known-candidate forward loss gradients, gate JVP and local Gram; no actual update','sources':{p:sha(R/p) for p in files},'blocking_findings':[],'probe_sha256':sha(O/'probe.py'),'probe_results_sha256':sha(O/'probe-results.json'),'limitations':['Y is physical unweighted and weights are fixed residual multipliers; caller provenance must establish that contract.','The unconstrained coupling gradient does not preserve rank-one couplings.','Gate Gram may be singular; no inverse, damping or update is implemented or approved.','Smooth analytic derivatives in FP64 are not certificates for rounded saturated bit patterns, descent, stationarity or generalization.','No evidence that gate-direction error caused prior quality failures; future actual protocol needs separate review.','Prior failures preserved; universal reconstruction unproved and private gate closed.']};(O/'source-review.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps(result));print('review_sha256',sha(O/'source-review.json'))
