"""Eight-row literal export-binding counterexample; no data IO/RNG/fit."""
from pathlib import Path
import sys,math,json
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');sys.path.insert(0,str(R/'examples/isolated_glu'))
import joint_response_audit_case as a
from joint_response_join import JointJob
from variable_response_view import ResponseBatch

def fixture():
 batches=[ResponseBatch(np.array([[1e20]]),np.array([[1.]])) for _ in range(8)]
 data=a._collect_fixture(JointJob('job00',1,1,1,8,iter(batches)),probes=4,rows=1)
 sigmoid=1/(1+math.exp(-1));records=[]
 for step in a.CHECKPOINTS:
  loss=(1-sigmoid)**2 if step==0 else 1.
  records.append({'step':step,'train_half_mse':loss/2,'guard_mse':loss,'guard_relative_error':1-sigmoid if step==0 else 1.,'selected_at_checkpoint':step==0,'selected_step_so_far':0})
 def checkpoint(step):return {'g':np.array([[1. if step==0 else 0.]]),'u':np.array([[1. if step==0 else 0.]]),'d':np.ones((1,1))}
 return dict(data=data,metadata=data.metadata(),records=records,selected_step=0,checkpoint=checkpoint,physical={'g':np.zeros((1,1)),'u':np.zeros((1,1)),'d':np.ones((1,1))},predictions=iter(np.zeros((2,1)) for _ in range(4)),saved_split={'train':data.train,'guard':data.guard},prediction_rows=2)

if __name__=='__main__':
 result=a.audit_case(**fixture());print(json.dumps({k:result[k] for k in ['status','physical_parameters_agreed','all_predictions_agreed','metrics_agreed','selection_certified']}))
