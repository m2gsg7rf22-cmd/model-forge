from pathlib import Path
import importlib.util, json, hashlib, ast
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
O=Path('/private/tmp/modelforge-bottleneck-review')
s=importlib.util.spec_from_file_location('bottleneck',R/'examples/isolated_glu/calibration_bottleneck.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rng=np.random.default_rng(3349);x=rng.normal(size=(19,5));g=rng.normal(size=(7,5));c=rng.normal(size=(7,3,5))
def direct(z):
 t=z@g.T
 return np.einsum('nw,wod,nd->no',t/(1+np.exp(-t)),c,z)
a=m.components(g,c,x)
parity=max(float(np.max(abs(a['even']-(direct(x)+direct(-x))/2))),float(np.max(abs(a['odd']-(direct(x)-direct(-x))/2))))
assert parity<1e-13
ref=np.einsum('nw,wod,nd->no',(x@g.T)**2,c,x)
assert np.allclose(ref,a['cubic'],rtol=1e-14,atol=1e-13)
tiny=m.components(g,c,x*1e-5)['odd']/(.25*1e-15)
err=float(np.linalg.norm(tiny-ref)/np.linalg.norm(ref));assert err<1e-8
p=m.canonical_probe_points(5);expected=np.vstack([np.eye(5)]+[(np.eye(5)[i]+np.eye(5)[j])[None,:] for i in range(5) for j in range(i+1,5)])
assert np.array_equal(p,expected)
tree=ast.parse((R/'examples/isolated_glu/calibration_bottleneck.py').read_text());imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
assert imports==['__future__','pathlib','typing','numpy.typing','modelforge.antipodal','modelforge.glu32_io']
result={'synthetic_parity_max_absolute':parity,'synthetic_cubic_limit_relative':err,'canonical_order_independently_verified':True,'real_saved_arrays_read':False,'queries':0,'fits':0,'optimizer_calls':0}
(O/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
