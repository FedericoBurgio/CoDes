from simulator import Solver
import utils

# Percorso del file URDF
urdf_path = "URDF/h2515.white.urdf"

solver = Solver(urdf_path, 'link6', True)

pos_traj, rot_traj, dt, T = utils.create_circular_traj([0.1,0.2,.9], .6, 10, 1.0, 1e-3)

qtraj, dqtraj, ddqtraj, eetraj, dt, T = solver.track_trajectory(pos_traj, rot_traj, dt, T)

solver.replay_traj(qtraj, eetraj)

utils.plot_traj(qtraj, dqtraj, ddqtraj, dt, T)
