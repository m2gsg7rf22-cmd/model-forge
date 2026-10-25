from pathlib import Path
import importlib.util,json
import numpy as np
repo=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');s=importlib.util.spec_from_file_location('sensitivity',repo/'examples/isolated_glu/residual_sensitivity.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
v=np.array([[1.,0,0],[-1.,0,0],[.9995,np.sqrt(1-.9995**2),0],[0,1,0],[0,0,1]]);losses=np.array([1e-9,1e-10,2e-9,1e-8,1.000001e-8]);before_v=v.copy();before_l=losses.copy();at=m.select(v,losses,1e-8);loose=m.select(v,losses,1e-7);assert at==[1,3] and loose==[1,3,4];assert np.array_equal(before_v,v) and np.array_equal(before_l,losses);assert len(m.select(np.eye(74),np.zeros(74),0))==74
# Greedy representatives need not equal connected components of a cosine graph.
angle=.03;lines=np.array([[1.,0],[np.cos(angle),np.sin(angle)],[np.cos(2*angle),np.sin(2*angle)]]);chain=m.select(lines,np.array([0.,1.,2.]),3);assert chain==[0,2]
assert m.select(v,losses,0)==[]
result={'strict_duplicate_boundary_absdot_equal_0_9995_rejected':True,'negative_sign_duplicate_uses_lowest_loss_representative':at[0]==1,'loss_equal_threshold_accepted':3 in at,'loss_just_above_threshold_excluded':4 not in at,'relaxed_control_selection_indices':loose,'original_arrays_unchanged':True,'no64_truncation_orthogonal74_retained':True,'greedy_not_transitive_equivalence_classes_control':chain,'empty_selection_valid':True,'no_real_telemetry_grid_inspected_optimizer_kernel_replay_queries_or_private_access':True};Path('/private/tmp/modelforge-sensitivity-review/probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
