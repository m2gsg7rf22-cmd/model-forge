from pathlib import Path
import importlib.util,json
import numpy as np
repo=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');spec=importlib.util.spec_from_file_location('v3',repo/'examples/isolated_glu/sealed_v3.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=np.random.default_rng(3392); d=rng.normal(size=(65,32));d/=np.linalg.norm(d,axis=1,keepdims=True);base={'directions':d[:63]};c=np.concatenate([np.tile(d[:63],(2,1)),d[63:64],d[63:64]])
a={'directions':d[:64].copy(),'all_directions':c.copy(),'residual':np.zeros(128)}
results={}
def check(name,mutate,basis,match):
    test={k:v.copy() for k,v in a.items()};mutate(test)
    try:m.audit_rescue_selection(base,test,basis)
    except ValueError as e:
        assert match in str(e),(name,str(e));results[name]=str(e)
    else:raise AssertionError(name+' was accepted')
m.audit_rescue_selection(base,a,np.eye(528));results['valid_synthetic']='accepted'
check('negative_squared_loss',lambda t:t['residual'].__setitem__(0,-1e-30),np.eye(528),'nonnegative')
check('zero_candidate',lambda t:t['all_directions'].__setitem__(0,0),np.eye(528),'norms')
check('reported_zero_actual_nonzero',lambda t:None,np.eye(528)[:128],'original subspace')
check('unlisted_retained_direction',lambda t:t['directions'].__setitem__(-1,d[64]),np.eye(528),'selection')
check('extra_65th_not_truncated',lambda t:t['all_directions'].__setitem__(-1,d[64]),np.eye(528),'selection')
print(json.dumps(results,indent=2));Path('/private/tmp/modelforge-v3-independent-review/probe-repaired-results.json').write_text(json.dumps(results,indent=2)+'\n')
