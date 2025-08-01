import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import time
import matplotlib.pyplot as plt

import meshcat.geometry as g
import meshcat.transformations as tf

import os


class Solver:
    def __init__(self, urdf_path: str, EE_name: str, rnd = False):
        # current_dir = os.path.dirname(os.path.abspath(__file__))

        # # Path to URDF file
        # urdf_path = os.path.join(current_dir, "..", "URDF", "h2515_white.urdf")

        # Temporarily change the working directory
        # urdf_folder = os.path.dirname(urdf_path)
        # prev_cwd = os.getcwd()  # Save current working directory
        
        # os.chdir(urdf_folder)
        # #self.urdf_path = urdf_path
        self.rmodel, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(urdf_path)
        # os.chdir(prev_cwd)
        
        self.data = self.rmodel.createData()
        self.end_effector = EE_name
        self.ee_id = self.rmodel.getFrameId(EE_name)
        self.maxExtension = -1
        #robot init
        self.robot_init(rnd)
        self.visual_init()

    def robot_init(self, rnd):
        self.q = np.empty(self.rmodel.nq)
        self.q[1]=np.pi # the base ahs an height, which is limiting (below 0?)
        self.dq = np.empty(self.rmodel.nq)
        pin.forwardKinematics(self.rmodel, self.data, 
                              self.q, self.dq)
        pin.updateFramePlacements(self.rmodel, self.data)
        self.maxExtension = np.linalg.norm(self.data.oMf[self.ee_id].translation)
        
        if rnd:
            self.q = np.random.uniform(-np.pi, np.pi, self.rmodel.nq)
            self.dq = np.random.uniform(-np.pi, np.pi, self.rmodel.nq)
            pin.forwardKinematics(self.rmodel, self.data, 
                              self.q, self.dq)
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
    
    def setpointRegulation(self, x_target = None, rot_target = None, 
                           kp_lin = 1e3, 
                           kd_lin = np.sqrt(2*1e4), 
                           kp_rot = 1e2,
                           kd_rot = np.sqrt(2*1e3)):
        ##target position
        if x_target == None:
            desired_pos = np.random.uniform(-1,1, 3)
            while np.linalg.norm(desired_pos) > 0.95*self.maxExtension:
                desired_pos = np.random.uniform(-1,1,3)    
        else:
            desired_pos = np.array(x_target)
        ##target orientation
        if  rot_target == None:
            rot_target = np.random.uniform(-np.pi, np.pi, 3)
        desired_rot = pin.rpy.rpyToMatrix(rot_target[0], rot_target[1], rot_target[2])

        ##trans matrix from origin to desired (target) pose
        oMfdesired = pin.SE3(desired_rot, desired_pos) #origin -> frameDesired (o_T_frameDesired)
        
        ##print pose
        import meshcat.geometry as g
        self.viz.viewer["point"].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0xff0000))
        self.viz.viewer["point"].set_transform(tf.translation_matrix(desired_pos))
        self.viz.viewer["targetPose"].set_object(g.triad(scale=0.2))
        self.viz.viewer["targetPose"].set_transform(oMfdesired.homogeneous)
        time.sleep(3)

        dt = 1e-3  # passo di integrazione
        total_time = 1.5
        steps = int(total_time / dt)

        Kp = np.diag([kp_lin]*3 + [kd_rot]*3)
        Kd = np.diag([kd_lin]*3 + [kd_rot]*3)

        q_traj = []
        dq_traj = []
        ddq_traj = []
        for i in range(steps):
            print(i)
            # Cinematica e dinamica
            
            #pin.forwardKinematics(rmodel, data, q, dq)
            #pin.updateFramePlacements(rmodel, data)
            #M = pin.crba(rmodel, data, q)
            #b = pin.rnea(rmodel, data, q, dq, np.zeros(rmodel.nv))
            #->
            pin.forwardKinematics(self.rmodel, self.data, self.q, self.dq)
            pin.updateFramePlacements(self.rmodel, self.data)
            pin.computeAllTerms(self.rmodel, self.data, self.q, self.dq)
            
            oMf=self.data.oMf[self.ee_id] #origin -> frame transformation
            fMfdesired = oMf.inverse() * oMfdesired # oMf^-1 = fMo; fMo * oMfdesired = fMfdesired. It is the transormation matrix from the current frame to the desired one, hence the error
            err = pin.log(fMfdesired).vector #log trasfroma da matrice a 6d, vector llo trasforma in numpy
            
            # Jacobian ee LOCAL
            J = pin.computeFrameJacobian(self.rmodel, self.data, self.q,
                self.ee_id,
                pin.ReferenceFrame.LOCAL)
            
            err_dx = - J @ self.dq
            ddx_des = Kp @ err + Kd @ err_dx
            
            ctrl = "osc"
            M = self.data.M.copy() #Mass matrix
            b = self.data.nle.copy() #Non Linear Effects (aka b. Coriolis, gravity)
            if ctrl == "osc":
                Lambda = np.linalg.inv(J @ np.linalg.inv(M) @ J.T + 1e-2 * np.eye(6))  #reg 1e-2 for stability
                mu     = Lambda @ (J @ np.linalg.inv(M) @ b) #non funziona !?
                f = Lambda.dot(ddx_des) + mu
            if ctrl == "pd":
                f = ddx_des
            if i == 1000: 
                print("forza")
                print(f[1])
                f[1]=1e3
                time.sleep(1)
            tau = J.T @ f 

            # Integrazione esplicita di Eulero
            #ddq = np.linalg.solve(M, tau - b)
            ddq=pin.aba(self.rmodel, self.data, self.q ,self.dq ,tau)
            self.dq += dt * ddq
            self.q = pin.integrate(self.rmodel, self.q ,dt * self.dq)
            
            q_traj.append(self.q.copy())
            dq_traj.append(self.dq.copy())
            ddq_traj.append(ddq.copy())
        return q_traj, dq_traj, ddq_traj, dt, total_time

    def create_circular_traj(self, center=np.array([.0, .0, 1.0]), radius=0.3, period=.2, z_height=1.0, dt=1e-3):
    
        total_time = period
        steps = int(total_time / dt)
        time_vec = np.linspace(0, total_time, steps)

        #position trajectory: start pos == final pos; 1 round
        #higher period slower tracking: closer points to follow with the same time step

        pos_traj = np.empty((steps, 3))
        for i, t in enumerate(time_vec): #i indice, t valore: t == time_vec[i] 
            x = center[0] + radius * np.cos(2 * np.pi * t / period)
            y = center[1] + radius * np.sin(2 * np.pi * t / period)
            z = z_height
            pos_traj[i] = np.array([x, y, z])
        
        # Orientation trajectory (fixed for simplicity, can be extended)
        rot_traj = np.repeat(np.eye(3)[np.newaxis, :, :], steps, axis=0)

        from meshcat.geometry import Line, PointsGeometry, PointsMaterial
        
        # Meshcat expects shape (3, N)
        pts = pos_traj.T  # shape (3, N)
        self.viz.viewer["fes_traj"].set_object(
            Line(
                PointsGeometry(pts),
                material=g.MeshLambertMaterial(color=0x00ff00, linewidth=12)
            )
        )

        return pos_traj, rot_traj, dt, total_time

    def track_trajectory(self, pos_traj, rot_traj, dt, total_time,
                    kp_lin=1e3, kd_lin=np.sqrt(2*1e4),
                    kp_rot=1e2, kd_rot=np.sqrt(2*1e3)):
        steps = pos_traj.shape[0]

        #preall or optimiz
        q_traj = np.empty((steps,self.rmodel.nq))
        dq_traj = np.empty((steps,self.rmodel.nq))
        ddq_traj = np.empty((steps,self.rmodel.nq))
        ee_traj = np.empty((steps,3)) #cartesian xyz, wrld frame

        for i in range(steps):
            desired_pos = pos_traj[i]
            desired_rot = rot_traj[i]

            oMfdesired = pin.SE3(desired_rot, desired_pos)

            # # Optional: visualize target in Meshcat
            # self.viz.viewer["point"].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0xff0000))
            # self.viz.viewer["point"].set_transform(tf.translation_matrix(desired_pos))
            # self.viz.viewer["targetPose"].set_object(g.triad(scale=0.2))
            # self.viz.viewer["targetPose"].set_transform(oMfdesired.homogeneous)
            # time.sleep(.01)
            pin.forwardKinematics(self.rmodel, self.data, self.q, self.dq)
            pin.updateFramePlacements(self.rmodel, self.data)
            pin.computeAllTerms(self.rmodel, self.data, self.q, self.dq)

            oMf = self.data.oMf[self.ee_id]
            
     
            fMfdesired = oMf.inverse() * oMfdesired
            err = pin.log(fMfdesired).vector

            J = pin.computeFrameJacobian(self.rmodel, self.data, self.q,
                                        self.ee_id, pin.ReferenceFrame.LOCAL)
            err_dx = - J @ self.dq

            # Gains
            Kp = np.diag([kp_lin]*3 + [kp_rot]*3)
            Kd = np.diag([kd_lin]*3 + [kd_rot]*3)
            ddx_des = Kp @ err + Kd @ err_dx

            # OSC (with regularization)
            M = self.data.M.copy()
            b = self.data.nle.copy()
            reg_eps = 1e-2
            Lambda = np.linalg.inv(J @ np.linalg.inv(M) @ J.T + reg_eps * np.eye(6))
            mu = Lambda @ (J @ np.linalg.inv(M) @ b)
            f = Lambda @ ddx_des + mu
            tau = J.T @ f  # No +b here, since mu alrdq_traj[]eady used

            # Integrate
            ddq = pin.aba(self.rmodel, self.data, self.q, self.dq, tau)
            self.dq += dt * ddq
            self.q = pin.integrate(self.rmodel, self.q, dt * self.dq)

            q_traj[i]=self.q.copy()
            dq_traj[i]=self.dq.copy()
            ddq_traj[i]=ddq.copy()
            ee_traj[i]=oMf.translation.copy()
            
            #ee_traj=np.vstack((ee_traj, oMf.translation.copy()))
            
            

        return q_traj, dq_traj, ddq_traj, ee_traj, dt, total_time

    def plot_traj(self, qTraj, dqTraj, ddqTraj, dt, total_time):
        time = np.arange(0,total_time,dt)

        plt.xlabel('Time')
        plt.ylabel('q')
        plt.plot(time, qTraj)
        plt.show()

        plt.xlabel('Time')
        plt.ylabel('dq')
        plt.plot(time, dqTraj)
        plt.show()

        plt.xlabel('Time')
        plt.ylabel('ddq')
        plt.plot(time, ddqTraj)
        plt.show()
 
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
        pts = pos_traj  # shape (3, N)
        self.viz.viewer[name].set_object(
            Line(
                PointsGeometry(pts),
                material=g.MeshLambertMaterial(color=color, linewidth=4)
            )
        )

    def replay_traj(self, qtraj, eetraj):
        j=0
        self.plot_taskspace_trajectory(eetraj, "eetraj")
        for q_step in qtraj:
            self.viz.display(q_step)
            time.sleep(1e-3)
            print(j)
            j+=1
   
# Percorso del file URDF
urdf_path = "URDF/h2515.white.urdf"
solver = Solver(urdf_path, 'link6', True)
pos_traj, rot_traj, dt, T = solver.create_circular_traj()

# Track it
qtraj, dqtraj, ddqtraj, eetraj, dt, T = solver.track_trajectory(pos_traj, rot_traj, dt, T)

# Plot and replay
#solver.plot_traj(qtraj, dqtraj, ddqtraj, dt, T)
solver.replay_traj(qtraj, eetraj)


x_target=[0.5,0,1]
rot_target = [0.2+np.pi/3,0.5,1]
qtraj, dqtraj, ddqtraj, dt, T = solver.setpointRegulation() #if None (default) is passed as either input, random x desired and/or rotation desired is randomly set
solver.plot_traj(qtraj, dqtraj, ddqtraj, dt, T)
solver.replay_traj(qtraj)
