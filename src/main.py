from simulator import Solver
import utils

# Percorso del file URDF
urdf_path = "URDF/h2515.white.urdf"

solver = Solver(urdf_path, 'link6', False)

pos_traj, rot_traj, dt, T = utils.create_circular_traj([0.1,0.2,.9], .6, 5, 1.0, 1e-3)

dt = 1e-3  # Time step for the simulation
T = 5  # Total time for the trajectory
pos_traj, rot_traj = utils.create_setpoint_traj(0.1, 0.2, 0.9, 0, 0, 0, T, dt)

qtraj, dqtraj, ddqtraj, eetraj, grinder_traj, dt, T = solver.track_trajectory(pos_traj, rot_traj, kp_lin=1000, kd_rot=100, dt=dt, total_time=T)

solver.replay_traj(qtraj, eetraj, grinder_traj)

utils.plot_traj(qtraj, dqtraj, ddqtraj, dt, T)
