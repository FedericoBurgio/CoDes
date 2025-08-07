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
    
        self.g = np.zeros(3)  #zeros for debuggging

        self.Kp = np.diag([1000, 1000, 1000, 100, 100, 100]) 
        self.Kd = np.diag([50, 50, 50, 10, 10, 10]) 

    def get_I_world(self):
        return self.R @ self.I_local @ self.R.T 

    def dyn(self, oMf, vEE, fext):
        #Iworld = self.R @ self.I_local @ self.R.T # fix this get wrld: you need the r from grinder to world frame
        Iworld = np.eye(3)  # For debugging, replace with actual inertia matrix if needed
        oRf = oMf.rotation
        xEE = oMf.translation
    
        deltax_lin = oRf @ (self.x[:3] - xEE[:3]) + (self.offsetEE)
        #deltax_rot = oRf @ (self.x[3:] - xEE[3:])  
    
        deltav_lin = oRf @ (self.xdot[:3] - vEE[:3])
        #deltav_rot = oRf @ (self.xdot[3:] - vEE[3:])

        # Compute the wrench in the world frame
        wrench = np.zeros(6)
        wrench[:3] = self.Kp[:3,:3] @ deltax_lin + self.Kd[:3,:3] @ deltav_lin
        #wrench[3:] = self.Kp[3:,3:] @ deltax_rot + self.Kd[3:,3:] @ deltav_rot
        #wrench[3:] = self.Kp[3:,3:] @ (self.xdot[3:] - vEE[3:]) + self.Kd[3:,3:] @ (self.xdotdot[3:] - vEE[3:])
        wrench[3:] = fext[3:]
        self.xdotdot[:3] = (-oRf @ wrench[:3] + fext[:3])
        

       
        #self.xdotdot[3:] = np.linalg.inv(Iworld) @ ((-oRf @ wrench[3:] + fext[3:]) - np.cross(self.xdot[3:], Iworld @ self.xdot[3:]))

        return wrench
    
    def step(self): #ONLY LINEAR FOR DEBUGGING
        self.xdot += self.xdotdot * self.dt
        self.x[:3] += self.xdot[:3] * self.dt
        print(f"Grinder position: {self.x[:3]}")  # Debugging print
        print(f"Grinder velocity: {self.xdot[:3]}")  # Debugging print
        print(f"Grinder acceleration: {self.xdotdot[:3]}")  # Debugging print


