#WIP not useed
import numpy as np
from controllers_payloads import RobotState, ControlCommand

class PDJointController:
    def __init__(self, rmodel, kp=1200.0):
        kd = np.sqrt(kp)
        self.Kp = np.diag([kp] * rmodel.nq)
        self.Kd = np.diag([kd] * rmodel.nq)

    def tau(self, state: RobotState, cmd: ControlCommand) -> np.ndarray:
        qd  = cmd.q_des  if cmd.q_des  is not None else np.zeros_like(state.q)
        dqd = cmd.dq_des if cmd.dq_des is not None else np.zeros_like(state.dq)
        return self.Kp @ (qd - state.q) + self.Kd @ (dqd - state.dq)
