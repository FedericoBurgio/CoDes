import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import time
import matplotlib.pyplot as plt

import meshcat.geometry as g
import meshcat.transformations as tf

import os
    
from grinder import GrinderAndInterface


class Solver:
    def __init__(self, urdf_path: str, EE_name: str, rnd = False):
  
        self.rmodel, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(urdf_path)
        self.data = self.rmodel.createData()
        self.end_effector = EE_name
        self.ee_id = self.rmodel.getFrameId(EE_name)
        self.maxExtension = -1

        #robot init
        self.robot_init(rnd)
        self.visual_init()

        #grinder - externally initialized
        self.grinder = GrinderAndInterface(5, np.eye(3), [0, 0, 0.05], self.data.oMf[self.ee_id])

    def robot_init(self, rnd):
        self.q = np.zeros(self.rmodel.nq)
        #self.q[1]=np.pi # the base ahs an height, which is limiting (below 0?)

        self.dq = np.zeros(self.rmodel.nq)
        pin.forwardKinematics(self.rmodel, self.data, self.q, self.dq)
        pin.updateFramePlacements(self.rmodel, self.data)
        self.maxExtension = np.linalg.norm(self.data.oMf[self.ee_id].translation)
        
        if rnd:
            self.q = np.random.uniform(-np.pi, np.pi, self.rmodel.nq)
            self.dq = np.random.uniform(-np.pi, np.pi, self.rmodel.nq)
            pin.forwardKinematics(self.rmodel, self.data, self.q, self.dq)
            pin.updateFramePlacements(self.rmodel, self.data)
 
    def visual_init(self):
        # Avvia MeshCat-server in un terminale a parte:
        # python -m meshcat.serving.server
        # Apri nel browser l'URL mostrato in console (es. http://127.0.0.1:7000/static/)
        self.viz = MeshcatVisualizer(self.rmodel, self.collision_model, self.visual_model)
        self.viz.initViewer(loadModel=True, open=True)
        self.viz.loadViewerModel()
        self.viz.display(self.q)
        time.sleep(2)
  
    def track_trajectory(self, pos_traj, rot_traj, dt, total_time,
                    kp_lin=1e3, kp_rot=1e2,
                    kd_lin=None, kd_rot=None):
        steps = pos_traj.shape[0]

        if kd_lin == None: kd_lin=np.sqrt(2*kp_lin)
        if kd_rot == None: kd_rot=np.sqrt(2*kp_rot)

        Kp = np.diag([kp_lin]*3 + [kp_rot]*3)
        Kd = np.diag([kd_lin]*3 + [kd_rot]*3)

        force_period = 0.2 #every .2 sec
        force_steps = int(force_period / dt) # dt = 1e-3 => every 200 steps 
    
        #preall or optimiz
        q_traj = np.empty((steps,self.rmodel.nq))
        dq_traj = np.empty((steps,self.rmodel.nq))
        ddq_traj = np.empty((steps,self.rmodel.nq))
        oMf_traj = [None] * steps #cartesian xyz, wrld frame

        grinder_traj = np.empty((steps,3)) #grinder position in world frame
        dqmax = np.array([3, 3, 3, 3, 3, 3]) #max joint velocity
        dqqmax = np.array([30, 30, 30, 30, 30, 30]) #max joint acceleration
        for i in range(steps):
            desired_pos = pos_traj[i]
            desired_rot = rot_traj[i]

            oMf = self.data.oMf[self.ee_id].copy()  # get current end-effector pose
            oRf = oMf.rotation  # Rotation of the end-effector in world frame
            oMfdesired = pin.SE3(desired_rot, desired_pos)
            fMfdesired = oMf.inverse() * oMfdesired
            err_local = pin.log(fMfdesired).vector
            
            E = np.block([[oRf, np.zeros((3,3))],
              [np.zeros((3,3)), oRf]])
            err = E @ err_local
            #err = np.zeros(6) # Dummy: EE stays in place (debugging)

            self.viz.viewer["Setpoint"].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0x0000ff))
            self.viz.viewer["Setpoint"].set_transform(tf.translation_matrix(pos_traj[i]))

            self.viz.viewer["SetPose"].set_object(g.triad(scale=0.2))
            self.viz.viewer["SetPose"].set_transform(pin.SE3(rot_traj[i], pos_traj[i]).homogeneous)

            self.viz.viewer["point"].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0xff0000))
            self.viz.viewer["point"].set_transform(tf.translation_matrix(oMf.translation))

            self.viz.viewer["pointGrinder"].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0x00ff00))
            self.viz.viewer["pointGrinder"].set_transform(tf.translation_matrix(self.grinder.x[:3]))
           
            vEE = pin.getFrameVelocity(self.rmodel, self.data, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
           
            #if i % force_steps == 0: f+=self.random_force(25)
            fext = self.random_force(50000)  if i % force_steps == 0 else np.zeros(6) #random force every force_period seconds
            #fext = np.array([1e4]*6) if i % force_steps == 0 else np.zeros(6)  # constant force for debugging
            #fext = np.array([0]*3+[1e5]+[0]*2) if i % force_steps == 0 else np.zeros(6)  # constant force for debugging
            wrench = self.grinder.dyn(oMf, vEE, fext) #get the wrench in the world frame

            #wrench = np.zeros(6) 

            pin.forwardKinematics(self.rmodel, self.data, self.q, self.dq)
            pin.updateFramePlacements(self.rmodel, self.data)
            pin.computeAllTerms(self.rmodel, self.data, self.q, self.dq)


            J = pin.computeFrameJacobian(self.rmodel, self.data, self.q,
                                        self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            err_dx = - J @ self.dq

            ddx_des = Kp @ err + Kd @ err_dx

            # OSC (with regularization)
            M = self.data.M.copy()
            b = self.data.nle.copy()
            reg_eps = 1e-4
            Lambda = np.linalg.inv(J @ np.linalg.inv(M) @ J.T + reg_eps * np.eye(6))
            mu = Lambda @ (J @ np.linalg.inv(M) @ b)
            f = Lambda @ ddx_des + mu - wrench

            #if i % force_steps == 0: f+=self.random_force(25) # apply f firectly to the ee

            tau = J.T @ f  # No +b here, since mu already used

            # Integrate
            ddq = np.clip(pin.aba(self.rmodel, self.data, self.q, self.dq, tau), -dqqmax, dqqmax)
            self.dq = np.clip(self.dq + ddq * dt, -dqmax, dqmax)
            self.q = pin.integrate(self.rmodel, self.q, dt * self.dq)
            
            self.grinder.step()
            
            q_traj[i]=self.q.copy()
            dq_traj[i]=self.dq.copy()
            ddq_traj[i]=ddq.copy()
            oMf_traj[i]=oMf.copy()

            grinder_traj[i] = self.grinder.x[:3].copy()
            
        return q_traj, dq_traj, ddq_traj, oMf_traj, grinder_traj, dt, total_time

    def random_force(self, force_std, torque_std=None):
        if torque_std is None:
            torque_std = 0.2 * force_std
        f = np.random.normal(0, force_std, 3)
        tau = np.random.normal(0, torque_std, 3)
        return np.concatenate([f, tau])

    def plot_taskspace_trajectory(self, pos_traj, name="desired_traj", color=0x00ff00):
        """
        Plot a trajectory in task space in Meshcat as a line.

        Args:
            pos_traj: np.ndarray of shape (N, 3), the positions (x, y, z)
            name: str, Meshcat object name
            color: int, RGB hex
        """
        from meshcat.geometry import Line, PointsGeometry, PointsMaterial
        # Meshcat expects shape (3, N)
        pts = pos_traj.T  # shape (3, N)
        self.viz.viewer[name].set_object(
            Line(
                PointsGeometry(pts),
                material=g.MeshLambertMaterial(color=color, linewidth=16)
            )
        )

    def replay_traj(self, qtraj, eetraj, grinder_traj):
        i=0
        import meshcat.geometry as g
        #breakpoint()
        while i < qtraj.shape[0]:
            self.viz.display(qtraj[i])
            desired_pos = eetraj[i].translation
            self.viz.viewer["targetPose"].set_object(g.triad(scale=0.2))
            self.viz.viewer["targetPose"].set_transform(eetraj[i].homogeneous)

            # Display grinder position
            self.viz.viewer["pointGrinder"].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0x00ff00))
            self.viz.viewer["pointGrinder"].set_transform(tf.translation_matrix(grinder_traj[i]))

            time.sleep(2e-3)
            print(i)
            i+=1
  