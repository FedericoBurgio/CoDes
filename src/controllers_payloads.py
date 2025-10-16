#WIP not useed
from dataclasses import dataclass
import numpy as np
import pinocchio as pin

@dataclass(frozen=True)
class RobotState:
    q: np.ndarray                 
    dq: np.ndarray              
    oMf_world: pin.SE3            
    J_lwa: np.ndarray             
    M: np.ndarray                 
    b: np.ndarray               

@dataclass(frozen=True)
class ControlCommand:
    # joint-space fields
    q_des: np.ndarray | None = None
    dq_des: np.ndarray | None = None
    # task-space fields
    x_des_world: np.ndarray | None = None      # (3,)
    R_des_world: np.ndarray | None = None      # (3,3)
