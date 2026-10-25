from pathlib import Path
import importlib.util,json
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-guard-normal-geometry-review');s=importlib.util.spec_from_file_location('error_geometry',R/'examples/isolated_glu/guard_normal_geometry.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);old=np.linalg.svd
try:
 def controlled(a,*args,**kwargs):
  if len(a)==2:raise np.linalg.LinAlgError('synthetic subset failure')
  return old(a,*args,**kwargs)
 np.linalg.svd=controlled;info=m.inspect_geometry(np.eye(3));assert len(info['subsets'])==8 and info['numerical_failures']==3 and not info['in_domain'];assert [v['mask'] for v in info['subsets'] if v['status']=='numerical_failure']==[3,5,6]
finally:np.linalg.svd=old
p=O/'probe-results.json';r=json.loads(p.read_text());r['all_subset_failures_retained_with_masks_verified']=True;p.write_text(json.dumps(r,indent=2)+'\n');print('synthetic SVD failure control passed; masks3,5,6 retained among all8')
