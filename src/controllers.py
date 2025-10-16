import numpy as np
import pinocchio as pin

class PDJointController:
    def __init__(self, rmodel, kp=1200.0):
        self.rmodel = rmodel
        self.kp = kp
        self.kd = np.sqrt(kp)
        self.Kp_joint = np.diag([self.kp] * rmodel.nq)
        self.Kd_joint = np.diag([self.kd] * rmodel.nq)

    def tau(self, q, dq, q_des, dq_des):
        return self.Kp_joint @ (np.asarray(q_des) - q) + self.Kd_joint @ (np.asarray(dq_des) - dq)

class OSCController: #operational space controller woek in progress NOT WORKING PROPERLY
    def __init__(self, rmodel, data, ee_id, kp_lin=1e3, kp_rot=1e2):
        self.rmodel = rmodel
        self.data = data
        self.ee_id = ee_id
        kd_lin = np.sqrt(kp_lin)
        kd_rot = np.sqrt(kp_rot)
        self.Kp = np.diag([kp_lin]*3 + [kp_rot]*3)
        self.Kd = np.diag([kd_lin]*3 + [kd_rot]*3)
        self.reg_eps = 1e-8

    def tau(self, q, dq, oMf, desired_pos, desired_rot):
        oRf = oMf.rotation
        oMfdesired = pin.SE3(desired_rot, desired_pos)
        fMfdesired = oMf.inverse() * oMfdesired
        err_local = pin.log(fMfdesired).vector
        E = np.block([[oRf, np.zeros((3, 3))], [np.zeros((3, 3)), oRf]])
        err = E @ err_local

        J = pin.computeFrameJacobian(self.rmodel, self.data, q, self.ee_id,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        err_dx = - J @ dq
        ddx_des = self.Kp @ err + self.Kd @ err_dx

        M = self.data.M.copy()
        b = self.data.nle.copy()
        Lambda = np.linalg.inv(J @ np.linalg.inv(M) @ J.T + self.reg_eps * np.eye(6))
        mu = Lambda @ (J @ np.linalg.inv(M) @ b)
        f = Lambda @ ddx_des + mu
        return J.T @ f