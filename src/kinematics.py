#WIP - not useed
import numpy as np
import pinocchio as pin

def damped_ls_ik(rmodel, data, ee_id, oMd, q_init, max_iters=50, tol=1e-4, damp=1e-3, step_size=1.0):
    q = q_init.copy()
    for _ in range(max_iters):
        pin.forwardKinematics(rmodel, data, q)
        pin.updateFramePlacements(rmodel, data)
        oMe = data.oMf[ee_id]
        eMe = oMe.inverse() * oMd
        err = pin.log(eMe).vector
        if np.linalg.norm(err) < tol:
            return q, True, _
        Jloc = pin.computeFrameJacobian(rmodel, data, q, ee_id, pin.ReferenceFrame.LOCAL)
        JJt = Jloc @ Jloc.T
        dq = Jloc.T @ np.linalg.solve(JJt + (damp ** 2) * np.eye(6), err)
        q = pin.integrate(rmodel, q, step_size * dq)
    return q, False, max_iters