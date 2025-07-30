import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import time

import meshcat.geometry as g
import meshcat.transformations as tf



class Solver:
    def __init__(self, urdf_path: str, EE_name: str, rnd = False):
        self.urdf_path = urdf_path
        self.rmodel, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(urdf_path)
        self.data = self.rmodel.createData()
        self.end_effector = EE_name
        self.ee_id = self.rmodel.getFrameId(EE_name)
        self.maxExtension = -1
        #robot init
        self.robot_init(rnd)
        self.visual_init()

    def robot_init(self, rnd):
        self.q = np.zeros(self.rmodel.nq)
        self.dq = np.zeros(self.rmodel.nq)
        pin.forwardKinematics(self.rmodel, self.data, 
                              self.q, self.dq)
        pin.updateFramePlacements(self.rmodel, self.data)
        self.maxExtension = np.linalg.norm(self.data.oMf[self.ee_id].translation)
        
        if rnd:
            self.q = np.random.uniform(-np.pi, np.pi, self.rmodel.nq)
            self.dq = np.random.uniform(-np.pi, -np.pi, self.rmodel.nq)
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
        #time.sleep(3)
    
    def setpointRegulation(self, x_target = None, rot_target = None, 
                           kp_lin = 1e4, 
                           kd_lin = np.sqrt(2*1e4), 
                           kp_rot = 1e3,
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
        total_time = 6
        steps = int(total_time / dt)

        Kp = np.diag([kp_lin]*3 + [kd_lin]*3)
        Kd = np.diag([kp_rot]*3 + [kd_rot]*3)

        trajectory = []

        for _ in range(steps):
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
                f = Lambda.dot(ddx_des) 
            if ctrl == "pd":
                f = ddx_des

            tau = J.T @ f + b

            # Integrazione esplicita di Eulero
            #ddq = np.linalg.solve(M, tau - b)
            ddq=pin.aba(self.rmodel, self.data, self.q ,self.dq ,tau)
            self.dq += dt * ddq
            #q  += dt * dq
            #dq = pin.integrate(rmodel, dq ,dt * ddq)
            self.q = pin.integrate(self.rmodel, self.q ,dt * self.dq)
            trajectory.append(self.q.copy())
        return trajectory

    def replay_traj(self, traj):
        for q_step in traj:
            self.viz.display(q_step)
            time.sleep(.01)

# Percorso del file URDF
urdf_path = "URDF/h2515.white.urdf"
solver = Solver(urdf_path, 'link6', True)
x_target=[0.5,0,1]
rot_target = [0.2+np.pi/3,0.5,1]
traj = solver.setpointRegulation() #if None (default) is passed as either input, random x desired and/or rotation desired is randomly set
solver.replay_traj(traj)
