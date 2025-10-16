import numpy as np
import pinocchio as pin

class GrinderAndInterface:
    def __init__(self, m, I_local, offset, k_lin, k_rot, oMf, dt): 
        self.m = m
        self.I_local = I_local
        self.dt = dt
        #offset is in EE frame
        self.offsetEE = offset.copy() #self.offset is in EE frame

        self.x = np.zeros(3) #position of the grinder in world frame
        self.xdot = np.zeros(3)  
        self.xdotdot = np.zeros(3) 

        self.theta = np.zeros(3)  # Angular velocity in world frame
        self.thetadot = np.zeros(3)  
        self.thetadotdot = np.zeros(3)  #

        self.x = oMf.act(np.array(self.offsetEE))  # Position in world frame
        self.R = oMf.rotation.copy()
    
        self.g = np.array([0, 0, -9.81])  # Gravity vector in world frame
      
        self.K = np.diag([k_lin]*3 + [k_rot]*3) #spring stiffness matrix (diag?)

        c_lin = np.sqrt(k_lin)
        c_rot = np.sqrt(k_rot)
        self.C = np.diag([c_lin]*3 + [c_rot]*3) #damping matrix
      
        # cache for step (set in dyn)
        self._x_nom_world = self.x.copy()
        self._r_off_world = oMf.rotation @ self.offsetEE

        self.debug_count = 0
    
    def dyn(self, oMf, xdotEE, fext_world, mext_world):
        #fext_world = np.asarray(fext_world).reshape(6,)
        self.debug_count += 1
        
        R = oMf.rotation           
        xEE = oMf.translation

        vEE = np.asarray(xdotEE.linear)
        wEE = np.asarray(xdotEE.angular)

        I_world = self.R @ self.I_local @ self.R.T 

        r_off_world = R @ self.offsetEE 
        x_nom_world = xEE + r_off_world

        ## PARTE LINEARE
        delta_x_EE  = R.T @ (self.x - x_nom_world)
        delta_v_EE = R.T @ (self.xdot - vEE)
        F_EE = - self.K[:3, :3] @ delta_x_EE - self.C[:3, :3] @ delta_v_EE
        F_world = R @ F_EE
        self.xdotdot = (F_world + fext_world) / self.m + self.g
        ##

        ## PARTE ROTAZIONALE
        delta_x_rot_EE = pin.log3(R.T @ self.R)
        delta_v_rot_EE = R.T @ (self.thetadot - wEE)
        T_EE    = - self.K[3:, 3:] @ delta_x_rot_EE - self.C[3:, 3:] @ delta_v_rot_EE
        T_world = R @ T_EE

        coriolis = np.cross(self.thetadot, I_world @ self.thetadot)   #self.xdot[3:] vel ang
        self.thetadotdot = np.linalg.solve(I_world, T_world + mext_world - coriolis) #T_world + fext_world[3:] (torque tot)
        ##


        #assemble the wrench
        f_wrench = F_world
        m_wrench = np.cross(r_off_world, F_world) + T_world #t_world+ np.cross to take into account the lever

        if self.debug_count < -1:
            breakpoint()
        return f_wrench, m_wrench, delta_x_EE

    def step(self):
        self.x += self.xdot * self.dt
        self.xdot += self.xdotdot * self.dt
        self.R = pin.exp3(self.thetadot * self.dt) @ self.R
        self.theta = pin.log3(self.R)
        self.thetadot += self.thetadotdot * self.dt
        

   