import numpy as np
import pinocchio as pin

class GrinderAndInterface:
    def __init__(self, m, I_local, offset, oMf): 
        self.m = m
        self.I_local = I_local
        self.dt = 1e-3
        #offset is in EE frame
        self.offsetEE = offset.copy() #self.offset is in EE frame

        self.x = np.zeros(6)
        self.xdot = np.zeros(6)  # Velocity of the grinder in world frame
        self.xdotdot = np.zeros(6)  # Acceleration of the grinder in world frame
        
        self.x[:3] = oMf.act(np.array(self.offsetEE))  # Position in world frame
        self.R = oMf.rotation.copy()
    
        self.g = np.array([0, 0, -9.81])  # Gravity vector in world frame
        #self.g = np.zeros(3)
        
        k_lin = 1e6
        k_rot = 1e5
        self.K = np.diag([k_lin]*3 + [k_rot]*3) #spring stiffness matrix (diag?)

        c_lin = 3e3
        c_rot = 1e2
        self.C = np.diag([c_lin]*3 + [c_rot]*3) #damping matrix
      
    def dynLIN(self, oMf, vEE_LWA, fext_world): #debugging
        R = oMf.rotation         
        xEE = oMf.translation

    
        r_off_world = R @ self.offsetEE
        x_nom_world = xEE + r_off_world

   
        delta_x_EE  = R.T @ (self.x[:3] - x_nom_world)
        delta_v_EE  = R.T @ (self.xdot[:3] - vEE_LWA.linear)  # LWA linear vel == world linear vel

        F_EE = - self.K[:3,:3] @ delta_x_EE - self.C[:3,:3] @ delta_v_EE

        F_world = R @ F_EE
        tau_world = np.cross(r_off_world, F_world)

        wrench = np.zeros(6)
        wrench[:3] = F_world
        wrench[3:] = tau_world

        self.xdotdot[:3] = (F_world + fext_world[:3]) / self.m + self.g

        return wrench
    
    def dyn(self, oMf, xdotEE_LWA, fext_world):
        #fext_world = np.asarray(fext_world).reshape(6,)
        
        R = oMf.rotation           
        xEE = oMf.translation

        vEE_LWA = np.asarray(xdotEE_LWA.linear)
        wEE_LWA = np.asarray(xdotEE_LWA.angular)

        I_world = self.R @ self.I_local @ self.R.T 

        r_off_world = R @ self.offsetEE 
        x_nom_world = xEE + r_off_world

        ## PARTE LINEARE
        delta_x_EE  = R.T @ (self.x[:3] - x_nom_world)
        delta_v_EE = R.T @ (self.xdot[:3] - vEE_LWA)

        F_EE = - self.K[:3, :3] @ delta_x_EE - self.C[:3, :3] @ delta_v_EE
        F_world = R @ F_EE

        self.xdotdot[:3] = (F_world + fext_world[:3]) / self.m + self.g
        ##


        ## PARTE ROTAZIONALE
        delta_x_rot_EE = pin.log3(R.T @ self.R)
        delta_v_rot_EE = R.T @ (self.xdot[3:] - wEE_LWA)
        T_EE    = - self.K[3:, 3:] @ delta_x_rot_EE - self.C[3:, 3:] @ delta_v_rot_EE
        T_world = R @ T_EE

        coriolis = np.cross(self.xdot[3:], I_world @ self.xdot[3:])   #self.xdot[3:] vel ang
        self.xdotdot[3:] = np.linalg.solve(I_world, T_world + fext_world[3:] - coriolis) #T_world + fext_world[3:] (torque tot)
        ##


        #assemble the wrench
        wrench = np.zeros(6)
        wrench[:3] = F_world
        wrench[3:] = np.cross(r_off_world, F_world) + T_world #t_world+ np.cross to take into account the lever
     
        return wrench

    def step(self):
        #Euler
        self.xdot += self.xdotdot * self.dt
        self.R = pin.exp3(self.xdot[3:] * self.dt) @ self.R
        self.x[:3] += self.xdot[:3] * self.dt
        self.x[3:] = pin.log3(self.R)

