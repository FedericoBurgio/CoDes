import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

def create_circular_traj(center, radius, period, z_height, dt):
    #2d ring
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

    #rot_traj = np.repeat(np.eye(3)[np.newaxis, :, :], steps, axis=0)
    
    rot_target = np.random.uniform(-np.pi, np.pi, 3)
    desired_rot = pin.rpy.rpyToMatrix(rot_target[0], rot_target[1], rot_target[2])
    rot_traj = np.repeat(desired_rot[np.newaxis, :, :], steps, axis=0)

    #plot_taskspace_trajectory(pos_traj, "des_traj")

    return pos_traj, rot_traj, dt, total_time

def create_setpoint_traj(x, y, z, xrot, yrot, zrot, T, dt):

    steps = int(T / dt)
    pos_traj = np.full((steps, 3), [x, y, z])
    desired_rot = pin.rpy.rpyToMatrix(xrot, yrot, zrot)
    rot_traj = np.repeat(desired_rot[np.newaxis, :, :], steps, axis=0)

    return pos_traj, rot_traj

def plot_traj(qTraj, dqTraj, ddqTraj, dt, total_time):
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

def position_tracking_error(pos_actual, pos_desired):
    """
    Computes per-step and aggregate position tracking errors.
    
    Args:
        pos_actual (np.ndarray): (steps, 3) array of actual positions (x, y, z)
        pos_desired (np.ndarray): (steps, 3) array of desired positions (x, y, z)
        
    Returns:
        errors (np.ndarray): (steps,) array of per-step Euclidean errors
        mean_error (float): mean error over all steps
        max_error (float): maximum error over all steps
        rmse (float): root mean square error
    """
    # Per-step Euclidean distance
    errors = np.linalg.norm(pos_actual - pos_desired, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    rmse = np.sqrt(np.mean(errors**2))
    return errors, mean_error, max_error, rmse
