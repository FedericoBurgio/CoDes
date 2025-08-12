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
        ##self.x[3:] = pin.rpy.matrixToRpy(oMf.rotation)
        self.R = oMf.rotation.copy()
    
        self.g = np.array([0, 0, -9.81])  # Gravity vector in world frame

        kp_lin = 5e3
        kp_rot = 5e2
        self.Kp = np.diag([kp_lin]*3 + [kp_rot]*3)
        
        kd_lin = np.sqrt(2 * kp_lin)
        kd_rot = np.sqrt(2 * kp_rot)
        self.Kd = np.diag([kd_lin]*3 + [kd_rot]*3)
   
    def get_I_world(self):
        return self.R @ self.I_local @ self.R.T 

    def dynOLD(self, oMf, vEE, fext):
        #Iworld = self.R @ self.I_local @ self.R.T # fix this get wrld: you need the r from grinder to world frame
        Iworld = np.eye(3)  # For debugging, replace with actual inertia matrix if needed
        oRf = oMf.rotation #likely wrong, maybe right in the case of the inertia matrix being np.eye(3)
        xEE = oMf.translation
        #self.x[:3] = oMf.act(np.array(self.offsetEE))
    
        deltax_lin = oRf @ (self.x[:3] - xEE[:3]) + self.offsetEE[:3]
        #deltax_rot = oRf @ (self.x[3:] - xEE[3:])  
    
        deltav_lin = oRf @ (self.xdot[:3] - vEE[:3])
        #deltav_rot = oRf @ (self.xdot[3:] - vEE[3:])

        # Compute the wrench in the world frame
        wrench = np.zeros(6)
        wrench[:3] = self.Kp[:3,:3] @ deltax_lin + self.Kd[:3,:3] @ deltav_lin
        #wrench[3:] = self.Kp[3:,3:] @ deltax_rot + self.Kd[3:,3:] @ deltav_rot
        #wrench[3:] = self.Kp[3:,3:] @ (self.xdot[3:] - vEE[3:]) + self.Kd[3:,3:] @ (self.xdotdot[3:] - vEE[3:])
        #wrench[3:] = fext[3:]
        self.xdotdot[:3] = self.g - (oRf.T @ wrench[:3] + fext[:3])/self.m
        

       
        #self.xdotdot[3:] = np.linalg.inv(Iworld) @ ((-oRf @ wrench[3:] + fext[3:]) - np.cross(self.xdot[3:], Iworld @ self.xdot[3:]))

        return wrench
    
    def dyn(self, oMf, vEE_LWA, fext_world):
        # EE pose/orientation
        R = oMf.rotation           # world->EE rotation is R^T; EE->world is R
        xEE = oMf.translation

        # Nominal mount point of grinder (world)
        r_off_world = R @ self.offsetEE
        x_nom_world = xEE + r_off_world

        # Deflection expressed in EE frame
        # Δx_EE = R^T (x - x_nom)
        delta_x_EE  = R.T @ (self.x[:3] - x_nom_world)
        delta_v_EE  = R.T @ (self.xdot[:3] - vEE_LWA[:3])  # LWA linear vel == world linear vel

        # Spring-damper force in EE frame
        F_EE = - self.Kp[:3,:3] @ delta_x_EE - self.Kd[:3,:3] @ delta_v_EE

        # Map to world/LWA
        F_world = R @ F_EE
        tau_world = np.cross(r_off_world, F_world)

        wrench = np.zeros(6)
        wrench[:3] = F_world
        wrench[3:] = tau_world

        self.xdotdot[:3] = (F_world + fext_world[:3]) / self.m + self.g

        # (Rotational dynamics omitted for now; keep R fixed)
        return wrench


    def step(self): #ONLY LINEAR FOR DEBUGGING
        self.xdot += self.xdotdot * self.dt
        self.x[:3] += self.xdot[:3] * self.dt
