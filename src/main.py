from simulator import Solver
import utils
import numpy as np
import json
import os
import argparse

urdf_path = "URDF/h2515.white.urdf"
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

conf = load_config("src/configs/conf1.json")
dt = conf["simulation"]["dt"]
T = conf["simulation"]["T"]


solver = Solver(urdf_path, 'link6', conf)

# pos_traj, rot_traj, dt, T = utils.create_circular_traj([0.1,0.2,.9], .6, 5, 1.0, dt)

pos_traj, rot_traj = utils.create_setpoint_traj(0, 0, 0, 0, 0, 0, T, dt)


qtraj, dqtraj, ddqtraj, tautraj, omftraj, grinder_traj, grinder_relative_traj, wrench_traj, deltax = (
    solver.follow_fixed_qdes()
)

if conf["grinder"]["grinder_in_use"]:
    exp_name = "grinder"
    if conf["forces"]["grinder_force"] != [0, 0, 0]:
        exp_name += f" F_grinder = {conf['forces']['grinder_force']}"
else:
    exp_name = "no grinder"

if conf["forces"]["EE_force"] != [0, 0, 0]:
    exp_name += f" F_EE = {conf['forces']['EE_force']}"


exp_name += f" T = {T}"
exp_name += f" dt = {dt}"


plotter = utils.ExperimentPlotter(
    set_name="pd_test_29_09_25",
    root_dir="experiments",
    experiment_name=exp_name,
)

plotter.plot_traj(qtraj, dqtraj, ddqtraj, dt, T, save=True, fname_prefix="joints")
plotter.plot_generic_data(
    tautraj, T, dt,
    title='Joint torques over time',
    legend=["t1","t2","t3","t4","t5","t6"],
    dataName='Torque [Nm]',
    save=True
)

# Visual replay

while 1:
    breakpoint()
    slowmo = 1
    solver.replay_traj(qtraj, omftraj, grinder_traj,slowmo)
