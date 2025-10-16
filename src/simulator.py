import os
import numpy as np
import time
from tqdm import tqdm
import pinocchio as pin

from grinder import GrinderAndInterface
from controllers import PDJointController, OSCController
from kinematics import damped_ls_ik
from visualization import maybe_create_viz

class Solver:
    def __init__(self, urdf_path: str, EE_name: str, conf):
        """Same public API; internal structure refactored."""
        self.rmodel, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(urdf_path)
        self.data = self.rmodel.createData()
        self.end_effector = EE_name
        self.ee_id = self.rmodel.getFrameId(EE_name)
        self.dt = conf["simulation"]["dt"]
        self.T = conf["simulation"]["T"]

        self.q_ik_seed = 0
        self.conf = conf    

        # robot + viz initialization
        self.robot_init()
        self.visual_init()
        self.grinder = None
        if self.conf["grinder"]["grinder_in_use"]:
            m = self.conf["grinder"]["mass"]
            I_local = np.eye(3)
            offset = np.array(self.conf["grinder"]["offset"])
            k_lin = self.conf["grinder"]["k_lin"]
            k_rot = self.conf["grinder"]["k_rot"]
            print("Grinder in use")
            self.grinder = GrinderAndInterface(m, I_local, offset, k_lin, k_rot, self.data.oMf[self.ee_id], self.dt)
        else:
            print("Grinder not in use")

    def robot_init(self):
        if self.conf["robot"]["random_init"]:
            self.q = np.random.uniform(-np.pi, np.pi, self.rmodel.nq)
            self.dq = np.random.uniform(-np.pi, np.pi, self.rmodel.nq)
            pin.forwardKinematics(self.rmodel, self.data, self.q, self.dq)
            pin.updateFramePlacements(self.rmodel, self.data)
        else:
            self.q = np.array(self.conf["robot"]["q_init"])
            self.dq = np.zeros(self.rmodel.nq)
            pin.forwardKinematics(self.rmodel, self.data, self.q, self.dq)
            pin.updateFramePlacements(self.rmodel, self.data)

    def visual_init(self):
        self.viz = maybe_create_viz(self.rmodel, self.collision_model, self.visual_model)
        if self.viz is not None:
            self.viz.display(self.q)
    
    def grinder_init(self, m, I_local, offset, k_lin, k_rot):
        self.grinder = GrinderAndInterface(m, I_local, offset, k_lin, k_rot, self.data.oMf[self.ee_id], self.dt)

    def solve_ik(self, oMd, q_init=None, max_iters=50, tol=1e-4, damp=1e-3, step_size=1.0):
        """Method name/signature preserved; delegates to helper."""
        if q_init is None:
            q_init = self.q.copy()
        return damped_ls_ik(self.rmodel, self.data, self.ee_id, oMd, q_init,
                            max_iters=max_iters, tol=tol, damp=damp, step_size=step_size)

    def follow_fixed_qdes(self):
        """Toggle with env vars for fast tests."""
        self.q_ik_seed = self.q.copy()
        steps = int(self.T/self.dt)
        pdj = PDJointController(self.rmodel, self.conf["pd"]["kp"])


        # prealloc
        q_traj = np.empty((steps, self.rmodel.nq))
        dq_traj = np.empty((steps, self.rmodel.nq))
        ddq_traj = np.empty((steps, self.rmodel.nq))
        wrench_traj = np.empty((steps, 6))
        deltax_traj = np.empty((steps, 3))
        tau_traj = np.empty((steps, 6))
        oMf_traj = [None] * steps
       
        grinder_traj = np.empty((steps, 3))
        grinder_relative_traj = np.empty((steps, 3))

        # limits 
        dqmax = np.array([3, 3, 3, 3, 3, 3])
        dqqmax = np.array([30, 30, 30, 30, 30, 30])

        rng = tqdm(range(0, steps))
        for i in rng:
            pin.forwardKinematics(self.rmodel, self.data, self.q, self.dq)
            pin.updateFramePlacements(self.rmodel, self.data)
            pin.computeAllTerms(self.rmodel, self.data, self.q, self.dq)

            oMf = self.data.oMf[self.ee_id].copy()
            vEE = pin.getFrameVelocity(self.rmodel, self.data, self.ee_id,
                                       pin.ReferenceFrame.WORLD)

            # external force 
            if i > self.conf["forces"]["EE_apply_after"]:
                f_world = np.array(self.conf["forces"]["EE_force"])
                m_world = np.array(self.conf["forces"]["EE_torque"])
            else:
                f_world = np.zeros(3)
                m_world = np.zeros(3)

            # grinder dynamics to wrench in world
            if self.grinder != None:
                if i > self.conf["forces"]["grinder_apply_after"] and i < self.conf["forces"]["grinder_force_release_after"]:
                    fextw = np.array(self.conf["forces"]["grinder_force"])
                    mextw = np.array(self.conf["forces"]["grinder_torque"])
                else:
                    fextw = np.zeros(3)
                    mextw = np.zeros(3)

                f_wrench, m_wrench, deltax = self.grinder.dyn(oMf, vEE, fextw, mextw)
                self.grinder.step()
            else:
                f_wrench = np.zeros(3)
                m_wrench = np.zeros(3)
                deltax = np.zeros(3)
  
            if self.conf["robot"]["q_target_is_q_init"]:
                q_des = self.conf["robot"]["q_init"]
            else:
                q_des = self.conf["robot"]["q_des"]
            
            dq_des = np.zeros_like(self.dq)
            tau_ctrl = pdj.tau(self.q, self.dq, q_des, dq_des)

            J = pin.computeFrameJacobian(self.rmodel, self.data, self.q, self.ee_id,
                                         pin.ReferenceFrame.WORLD)
            

            Jlin = J[:3, :]   # linear rows (vx, vy, vz)
            Jang = J[3:, :]   # angular rows (wx, wy, wz)

            tau_ext    = Jlin.T @ f_world   + Jang.T @ m_world
            tau_wrench = Jlin.T @ f_wrench  + Jang.T @ m_wrench
            tau_cmd    = tau_ctrl - tau_ext + tau_wrench
            
        
            ddq = pin.aba(self.rmodel, self.data, self.q, self.dq, tau_cmd)
            self.q = pin.integrate(self.rmodel, self.q, self.dt * self.dq)
            self.dq = self.dq + ddq * self.dt

            if self.grinder is not None:
                self.grinder.step()

            q_traj[i] = self.q.copy()
            dq_traj[i] = self.dq.copy()
            ddq_traj[i] = ddq.copy()
            oMf_traj[i] = oMf.copy()
            oRf = oMf.rotation
            if self.grinder is not None:
                grinder_traj[i] = self.grinder.x[:3].copy()
                grinder_relative_traj[i] = oRf.T @ (self.grinder.x[:3] - oMf.translation)
                wrench_traj[i] = np.hstack([f_wrench, m_wrench])
            deltax_traj[i] = deltax.copy()
            tau_traj[i] = tau_cmd.copy()

            rng.set_description(f"Sim: {i+1}/{steps})")


        return (
            q_traj, dq_traj, ddq_traj, tau_traj, oMf_traj,
            grinder_traj, grinder_relative_traj, wrench_traj, deltax_traj
        )


    def random_force(self, force_std, torque_std=None):
        if torque_std is None:
            torque_std = 0.2 * force_std
        f = np.random.normal(0, force_std, 3)
        tau = np.random.normal(0, torque_std, 3)
        return np.concatenate([f, tau])

    def plot_taskspace_trajectory(self, pos_traj, name="desired_traj", color=0x00ff00):
        from meshcat.geometry import Line, PointsGeometry
        import meshcat.geometry as g
        if self.viz is None:
            return
        pts = pos_traj.T
        self.viz.viewer[name].set_object(
            Line(PointsGeometry(pts), material=g.MeshLambertMaterial(color=color, linewidth=16))
        )

    def replay_traj(self, qtraj, eetraj, grinder_traj, slowmo=60):
        # Method name preserved; behavior tightened for 60Hz-ish playback
        if self.viz is None:
            return
        import meshcat.geometry as g
        import meshcat.transformations as tf
        i = 0
        inc = max(1, int((1 / self.dt) / 60))
        inc = int(inc/slowmo)
        #sleep_time = 1 / fps
        bar = tqdm(total=qtraj.shape[0])
        while i < qtraj.shape[0]:
            self.viz.display(qtraj[i])
            self.viz.viewer["targetPose"].set_object(g.triad(scale=0.2))
            self.viz.viewer["targetPose"].set_transform(eetraj[i].homogeneous)
            if self.grinder is not None:
                self.viz.viewer["pointGrinder"].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0x00ff00))
                self.viz.viewer["pointGrinder"].set_transform(tf.translation_matrix(grinder_traj[i]))
            self.viz.viewer["EEpoint"].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0xff0000))
            self.viz.viewer["EEpoint"].set_transform(tf.translation_matrix(eetraj[i].translation))
            i += inc
            bar.update(inc)
            time.sleep(.01666)
        bar.close()