# import pinocchio as pin
# from pinocchio.visualize import MeshcatVisualizer
# import numpy as np
# import time
# import matplotlib.pyplot as plt

# import meshcat.geometry as g
# import meshcat.transformations as tf

# import os
    

# class GrinderAndInterface:#in grinder.py
#     def __init__(self, m, I_local, offset, R0):
#         self.m = m
#         self.I_local = I_local
#         self.offset = offset.copy()  # Ensure offset is a copy to avoid modifying the original
#         self.R = R0.copy()
        
#         self.x = np.zeros(6)  # Position of the grinder in world frame
#         self.x[:3] = np.array(offset.copy())
#         #self.x[3:]

#         self.xdot = np.zeros(6)  # Velocity of the grinder in world frame
#         self.xdotdot = np.zeros(6)  # Acceleration of the grinder in world frame

#         self.g = np.array([0, 0, -9.81])
#         self.g = np.zeros(3)  
#         #self.state = (self.x, self.R, self.v, self.omega)  # (position, rotation, velocity, angular velocity)

#         self.Kp = np.diag([1000, 1000, 1000, 100, 100, 100]) #position and orientation
#         self.Kd = np.diag([50, 50, 50, 10, 10, 10]) #velocity and angular velocity
        
#         self.I_world = self.get_I_world()  # Inertia in world frame #FIXARE R

#     def get_I_world(self):
#         return self.R @ self.I_local @ self.R.T #fix R

#     def dyn(self, oRf, xEE, vEE, fext):
#         Iworld = self.R @ self.I_local @ self.R.T # fix this get wrld: you need the r from grinder to world frame
     
#         deltax_lin = oRf @ (self.x[:3] - xEE[:3]) + (self.offset)
#         deltax_rot = oRf @ (self.x[3:] - xEE[3:])  
    
#         deltav_lin = oRf @ (self.xdot[:3] - vEE[:3])
#         deltav_rot = oRf @ (self.xdot[3:] - vEE[3:])

#         # Compute the wrench in the world frame
#         wrench = np.empty(6)
#         wrench[:3] = self.Kp[:3,:3] @ deltax_lin + self.Kd[:3,:3] @ deltav_lin
#         wrench[3:] = self.Kp[3:,3:] @ deltax_rot + self.Kd[3:,3:] @ deltav_rot

        
#         self.xdotdot[:3] = self.g @ (-oRf @ wrench[:3] + fext[:3])
       
#         self.xdotdot[3:] = np.linalg.inv(Iworld) @ ((-oRf @ wrench[3:] + fext[3:]) - np.cross(self.xdot[3:], Iworld @ self.xdot[3:]))

#         return wrench

# class Solver:
#     def __init__(self, urdf_path: str, EE_name: str, rnd = False):
  
#         self.rmodel, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(urdf_path)
#         self.data = self.rmodel.createData()
#         self.end_effector = EE_name
#         self.ee_id = self.rmodel.getFrameId(EE_name)
#         self.maxExtension = -1

#         #robot init
#         self.robot_init(rnd)
#         self.visual_init()

#         #grinder - externally initialized
#         self.grinder = GrinderAndInterface(10, np.eye(3), [.2,.2,0], self.data.oMf[self.ee_id].rotation)

#     def robot_init(self, rnd):
#         self.q = np.empty(self.rmodel.nq)
#         self.q[1]=np.pi # the base ahs an height, which is limiting (below 0?)
#         self.dq = np.empty(self.rmodel.nq)
#         pin.forwardKinematics(self.rmodel, self.data, 
#                               self.q, self.dq)
#         pin.updateFramePlacements(self.rmodel, self.data)
#         self.maxExtension = np.linalg.norm(self.data.oMf[self.ee_id].translation)
        
#         if rnd:
#             self.q = np.random.uniform(-np.pi, np.pi, self.rmodel.nq)
#             self.dq = np.random.uniform(-np.pi, np.pi, self.rmodel.nq)
#             pin.forwardKinematics(self.rmodel, self.data, 
#                               self.q, self.dq)
#             pin.updateFramePlacements(self.rmodel, self.data)
 
#     def visual_init(self):
#         # Avvia MeshCat-server in un terminale a parte:
#         # python -m meshcat.serving.server
#         # Apri nel browser l'URL mostrato in console (es. http://127.0.0.1:7000/static/)
#         self.viz = MeshcatVisualizer(self.rmodel, self.collision_model, self.visual_model)
#         self.viz.initViewer(loadModel=True, open=True)
#         self.viz.loadViewerModel()
#         self.viz.display(self.q)
#         time.sleep(2)
    
#     def setpointRegulation(self, x_target = None, rot_target = None, 
#                            kp_lin = 1e3, 
#                            kd_lin = np.sqrt(2*1e3), 
#                            kp_rot = 1e2,
#                            kd_rot = np.sqrt(2*1e2)):
#         ##target position
#         if x_target == None:
#             desired_pos = np.random.uniform(-1,1, 3)
#             while np.linalg.norm(desired_pos) > 0.95*self.maxExtension:
#                 desired_pos = np.random.uniform(-1,1,3)    
#         else:
#             desired_pos = np.array(x_target)
#         ##target orientation
#         if  rot_target == None:
#             rot_target = np.random.uniform(-np.pi, np.pi, 3)
#         desired_rot = pin.rpy.rpyToMatrix(rot_target[0], rot_target[1], rot_target[2])

#         ##trans matrix from origin to desired (target) pose
#         oMfdesired = pin.SE3(desired_rot, desired_pos) #origin -> frameDesired (o_T_frameDesired)
        
#         ##print pose
#         import meshcat.geometry as g
#         self.viz.viewer["point"].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0xff0000))
#         self.viz.viewer["point"].set_transform(tf.translation_matrix(desired_pos))
#         self.viz.viewer["targetPose"].set_object(g.triad(scale=0.2))
#         self.viz.viewer["targetPose"].set_transform(oMfdesired.homogeneous)
#         time.sleep(3)

#         dt = 1e-3  # passo di integrazione
#         total_time = 1.5
#         steps = int(total_time / dt)

#         Kp = np.diag([kp_lin]*3 + [kd_rot]*3)
#         Kd = np.diag([kd_lin]*3 + [kd_rot]*3)

#         EE_int_wrench = self.grinder.dyn(self.rmodel, self.data, self.q, self.dq)

#         q_traj = []
#         dq_traj = []
#         ddq_traj = []
#         for i in range(steps):
#             print(i)
#             # Cinematica e dinamica
            
#             #pin.forwardKinematics(rmodel, data, q, dq)
#             #pin.updateFramePlacements(rmodel, data)
#             #M = pin.crba(rmodel, data, q)
#             #b = pin.rnea(rmodel, data, q, dq, np.zeros(rmodel.nv))
#             #->
#             pin.forwardKinematics(self.rmodel, self.data, self.q, self.dq)
#             pin.updateFramePlacements(self.rmodel, self.data)
#             pin.computeAllTerms(self.rmodel, self.data, self.q, self.dq)
            
#             oMf=self.data.oMf[self.ee_id] #origin -> frame transformation
#             fMfdesired = oMf.inverse() * oMfdesired # oMf^-1 = fMo; fMo * oMfdesired = fMfdesired. It is the transormation matrix from the current frame to the desired one, hence the error
#             err = pin.log(fMfdesired).vector #log trasfroma da matrice a 6d, vector llo trasforma in numpy
            
#             # Jacobian ee LOCAL
#             J = pin.computeFrameJacobian(self.rmodel, self.data, self.q,
#                 self.ee_id,
#                 pin.ReferenceFrame.LOCAL)
            
#             err_dx = - J @ self.dq
#             ddx_des = Kp @ err + Kd @ err_dx
            
#             ctrl = "osc"
#             M = self.data.M.copy() #Mass matrix
#             b = self.data.nle.copy() #Non Linear Effects (aka b. Coriolis, gravity)
#             if ctrl == "osc":
#                 Lambda = np.linalg.inv(J @ np.linalg.inv(M) @ J.T + 1e-2 * np.eye(6))  #reg 1e-2 for stability
#                 mu     = Lambda @ (J @ np.linalg.inv(M) @ b) #non funziona !?
#                 f = Lambda.dot(ddx_des) + mu
#             if ctrl == "pd":
#                 f = ddx_des
#             if i == 1000: 
#                 print("forza")
#                 print(f[1])
#                 f[1]=1e3
#                 time.sleep(1)
#             tau = J.T @ f 

#             # Integrazione esplicita di Eulero
#             #ddq = np.linalg.solve(M, tau - b)
#             ddq=pin.aba(self.rmodel, self.data, self.q ,self.dq ,tau)
#             self.dq += dt * ddq
#             self.q = pin.integrate(self.rmodel, self.q ,dt * self.dq)
            
#             q_traj.append(self.q.copy())
#             dq_traj.append(self.dq.copy())
#             ddq_traj.append(ddq.copy())
#         return q_traj, dq_traj, ddq_traj, dt, total_time

#     def create_circular_traj(self, center=np.array([.0, .0, 1.0]), radius=0.5, period=2, z_height=1.0, dt=1e-3):##utils
    
#         total_time = period
#         steps = int(total_time / dt)
#         time_vec = np.linspace(0, total_time, steps)

#         #position trajectory: start pos == final pos; 1 round
#         #higher period slower tracking: closer points to follow with the same time step

#         pos_traj = np.empty((steps, 3))
        
#         for i, t in enumerate(time_vec): #i indice, t valore: t == time_vec[i] 
#             x = center[0] + radius * np.cos(2 * np.pi * t / period)
#             y = center[1] + radius * np.sin(2 * np.pi * t / period)
#             z = z_height
#             pos_traj[i] = np.array([x, y, z])
        
#         # Orientation trajectory (fixed for simplicity, can be extended)
#         #rot_traj = np.repeat(np.eye(3)[np.newaxis, :, :], steps, axis=0)
#         rot_target = np.random.uniform(-np.pi, np.pi, 3)
#         desired_rot = pin.rpy.rpyToMatrix(rot_target[0], rot_target[1], rot_target[2])
#         rot_traj = np.repeat(desired_rot[np.newaxis, :, :], steps, axis=0)

#         self.plot_taskspace_trajectory(pos_traj, "des_traj")

#         return pos_traj, rot_traj, dt, total_time

#     def track_trajectory(self, pos_traj, rot_traj, dt, total_time,
#                     kp_lin=1e3, kp_rot=None,
#                     kd_lin=1e2, kd_rot=None):
#         steps = pos_traj.shape[0]

#         if kd_rot == None: kd_rot=np.sqrt(2*kd_lin)
#         if kp_rot == None: kp_rot=np.sqrt(2*kp_lin)
#         Kp = np.diag([kp_lin]*3 + [kp_rot]*3)
#         Kd = np.diag([kd_lin]*3 + [kd_rot]*3)

#         force_period = 0.2 #every .2 sec
#         force_steps = int(force_period / dt) # dt = 1e-3 => every 200 steps 
        
#         # rot_target = np.random.uniform(-np.pi, np.pi, 3)
#         # desired_rot = pin.rpy.rpyToMatrix(rot_target[0], rot_target[1], rot_target[2])
#         # breakpoint()

        
#         #preall or optimiz
#         q_traj = np.empty((steps,self.rmodel.nq))
#         dq_traj = np.empty((steps,self.rmodel.nq))
#         ddq_traj = np.empty((steps,self.rmodel.nq))
#         ee_traj = np.empty((steps,3)) #cartesian xyz, wrld frame

#         for i in range(steps):
#             desired_pos = pos_traj[i]
#             desired_rot = rot_traj[i]

#             oMfdesired = pin.SE3(desired_rot, desired_pos)

#             # # Optional: visualize target in Meshcat
#             # self.viz.viewer["point"].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0xff0000))
#             # self.viz.viewer["point"].set_transform(tf.translation_matrix(desired_pos))
#             # self.viz.viewer["targetPose"].set_object(g.triad(scale=0.2))
#             # self.viz.viewer["targetPose"].set_transform(oMfdesired.homogeneous)
#             # time.sleep(.01)

            
#             xEE = pin.log(self.data.oMf[self.ee_id]).vector  # get end-effector position in world frame
#             # orientation: convert to minimal 3D (or keep rotation matrix)
#             # get local orientation error if using angle-axis
#             # xEE[3:] = orientation_from_rotation_matrix(oMf.rotation)
#             # vEE = pin.getFrameVelocity(self.rmodel, self.data, self.ee_id, pin.ReferenceFrame.WORLD).vector
#             vEE = pin.getFrameVelocity(self.rmodel, self.data, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector

#             #if i % force_steps == 0: f+=self.random_force(25)
#             fext = self.random_force(25)  if i % force_steps == 0 else np.zeros(6) #random force every force_period seconds
#             wrench = self.grinder.dyn(oMfdesired.rotation, xEE, vEE, np.zeros(6)) #get the wrench in the world frame   
            
#             #wrench = np.zeros(6) 
            
#             self.grinder.R = self.data.oMf[self.ee_id].rotation @ pin.exp3(self.grinder.xdotdot[3:]*dt)
#             self.grinder.xdot += self.grinder.xdotdot * dt
#             self.grinder.x += self.grinder.xdot * dt

#             pin.forwardKinematics(self.rmodel, self.data, self.q, self.dq)
#             pin.updateFramePlacements(self.rmodel, self.data)
#             pin.computeAllTerms(self.rmodel, self.data, self.q, self.dq)

#             oMf = self.data.oMf[self.ee_id]
#             fMfdesired = oMf.inverse() * oMfdesired
#             err = pin.log(fMfdesired).vector

#             J = pin.computeFrameJacobian(self.rmodel, self.data, self.q,
#                                         self.ee_id, pin.ReferenceFrame.LOCAL)
#             err_dx = - J @ self.dq

#             ddx_des = Kp @ err + Kd @ err_dx

#             # OSC (with regularization)
#             M = self.data.M.copy()
#             b = self.data.nle.copy()
#             reg_eps = 1e-4
#             Lambda = np.linalg.inv(J @ np.linalg.inv(M) @ J.T + reg_eps * np.eye(6))
#             mu = Lambda @ (J @ np.linalg.inv(M) @ b)
#             f = Lambda @ ddx_des + mu - wrench

#             #if i % force_steps == 0: f+=self.random_force(25)

#             tau = J.T @ f  # No +b here, since mu alrdq_traj[]eady used

#             # Integrate
#             ddq = pin.aba(self.rmodel, self.data, self.q, self.dq, tau)
#             self.dq += dt * ddq
#             self.q = pin.integrate(self.rmodel, self.q, dt * self.dq)

#             q_traj[i]=self.q.copy()
#             dq_traj[i]=self.dq.copy()
#             ddq_traj[i]=ddq.copy()
#             ee_traj[i]=oMf.translation.copy()
            
#             #ee_traj=np.vstack((ee_traj, oMf.translation.copy()))
            
            

#         return q_traj, dq_traj, ddq_traj, ee_traj, dt, total_time

#     def random_force(self, force_std, torque_std = None):
#         if torque_std == None: torque_std = force_std * 0.2
#         return np.concatenate([np.random.normal(0, force_std, 3),np.random.normal(0, force_std, 3)])

#     def plot_traj(self, qTraj, dqTraj, ddqTraj, dt, total_time):##utils
#         time = np.arange(0,total_time,dt)

#         plt.xlabel('Time')
#         plt.ylabel('q')
#         plt.plot(time, qTraj)
#         plt.show()

#         plt.xlabel('Time')
#         plt.ylabel('dq')
#         plt.plot(time, dqTraj)
#         plt.show()

#         plt.xlabel('Time')
#         plt.ylabel('ddq')
#         plt.plot(time, ddqTraj)
#         plt.show()
 
#     def plot_taskspace_trajectory(self, pos_traj, name="desired_traj", color=0x00ff00):
#         """
#         Plot a trajectory in task space in Meshcat as a line.

#         Args:
#             pos_traj: np.ndarray of shape (N, 3), the positions (x, y, z)
#             name: str, Meshcat object name
#             color: int, RGB hex
#         """
#         from meshcat.geometry import Line, PointsGeometry, PointsMaterial
#         # Meshcat expects shape (3, N)
#         pts = pos_traj.T  # shape (3, N)
#         self.viz.viewer[name].set_object(
#             Line(
#                 PointsGeometry(pts),
#                 material=g.MeshLambertMaterial(color=color, linewidth=16)
#             )
#         )

#     def replay_traj(self, qtraj, eetraj):
#         j=0
#         self.plot_taskspace_trajectory(eetraj, "eetraj", color=0xff0000)
#         for q_step in qtraj:
#             self.viz.display(q_step)
#             time.sleep(1e-3)
#             print(j)
#             j+=1

# def position_tracking_error(pos_actual, pos_desired):##utils
#     """
#     Computes per-step and aggregate position tracking errors.
    
#     Args:
#         pos_actual (np.ndarray): (steps, 3) array of actual positions (x, y, z)
#         pos_desired (np.ndarray): (steps, 3) array of desired positions (x, y, z)
        
#     Returns:
#         errors (np.ndarray): (steps,) array of per-step Euclidean errors
#         mean_error (float): mean error over all steps
#         max_error (float): maximum error over all steps
#         rmse (float): root mean square error
#     """
#     # Per-step Euclidean distance
#     errors = np.linalg.norm(pos_actual - pos_desired, axis=1)
#     mean_error = np.mean(errors)
#     max_error = np.max(errors)
#     rmse = np.sqrt(np.mean(errors**2))
#     return errors, mean_error, max_error, rmse

# # Percorso del file URDF
# urdf_path = "URDF/h2515.white.urdf"

# solver = Solver(urdf_path, 'link6', True)

# pos_traj, rot_traj, dt, T = solver.create_circular_traj()

# # Track it
# qtraj, dqtraj, ddqtraj, eetraj, dt, T = solver.track_trajectory(pos_traj, rot_traj, dt, T)

# # Plot and replay
# #solver.plot_traj(qtraj, dqtraj, ddqtraj, dt, T)
# solver.replay_traj(qtraj, eetraj)


# # x_target=[0.5,0,1]
# # rot_target = [0.2+np.pi/3,0.5,1]
# # qtraj, dqtraj, ddqtraj, dt, T = solver.setpointRegulation() #if None (default) is passed as either input, random x desired and/or rotation desired is randomly set
# # solver.plot_traj(qtraj, dqtraj, ddqtraj, dt, T)
# # solver.replay_traj(qtraj)
