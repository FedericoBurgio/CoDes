import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Sequence

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt


def create_circular_traj(center, radius, period, z_height, dt):
    total_time = period
    steps = int(total_time / dt)
    time_vec = np.linspace(0, total_time, steps)

    pos_traj = np.empty((steps, 3))
    for i, t in enumerate(time_vec):
        x = center[0] + radius * np.cos(2 * np.pi * t / period)
        y = center[1] + radius * np.sin(2 * np.pi * t / period)
        z = z_height
        pos_traj[i] = np.array([x, y, z])

    rot_target = np.random.uniform(-np.pi, np.pi, 3)
    desired_rot = pin.rpy.rpyToMatrix(rot_target[0], rot_target[1], rot_target[2])
    rot_traj = np.repeat(desired_rot[np.newaxis, :, :], steps, axis=0)

    return pos_traj, rot_traj, dt, total_time


def create_setpoint_traj(x, y, z, xrot, yrot, zrot, T, dt): #series of identical points
    steps = int(T / dt)
    pos_traj = np.full((steps, 3), [x, y, z])
    desired_rot = pin.rpy.rpyToMatrix(xrot, yrot, zrot)
    rot_traj = np.repeat(desired_rot[np.newaxis, :, :], steps, axis=0)
    return pos_traj, rot_traj


def _slugify(text: str) -> str:
    return text
    # text = text.strip().lower()
    # text = re.sub(r"[^\w\s-]", "", text)
    # text = re.sub(r"[\s_-]+", "-", text)
    # return text.strip("-")


class ExperimentPlotter:
    """Organizes plots by experiment set and saves figures if requested."""
    def __init__(self, set_name: str, root_dir: str = "experiments", experiment_name: Optional[str] = None, timestamp: Optional[str] = None):
        set_slug = _slugify(set_name)
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_slug = _slugify(experiment_name) + "_" + ts if experiment_name else f"exp_{ts}"
        self.base_dir = Path(root_dir) / f"set_{set_slug}" / exp_slug
        self.plots_dir = self.base_dir / "plots"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self._write_meta({
            "set_name": set_name,
            "experiment_name": experiment_name,
            "timestamp": ts,
            "root_dir": str(self.base_dir)
        })

    def plot_traj(self, qTraj: np.ndarray, dqTraj: np.ndarray, ddqTraj: np.ndarray, dt: float,
                  total_time: Optional[float] = None, save: bool = True, fname_prefix: Optional[str] = None,
                  show: bool = True, dpi: int = 150):
        self._plot_single(qTraj, dt, total_time, "Joint position", "q", "Time", save, self._fname(fname_prefix, "q"), show, dpi)
        self._plot_single(dqTraj, dt, total_time, "Joint velocity", "dq", "Time", save, self._fname(fname_prefix, "dq"), show, dpi)
        self._plot_single(ddqTraj, dt, total_time, "Joint acceleration", "ddq", "Time", save, self._fname(fname_prefix, "ddq"), show, dpi)

    def plot_generic_data(self, data: np.ndarray, total_time: float, dt: float, title: str = "",
                          legend: Sequence[str] | str = (), dataName: str = "data", save: bool = True,
                          filename: Optional[str] = None, show: bool = True, dpi: int = 150):
        time = self._time_vector(data.shape[0], dt, total_time)
        fig, ax = plt.subplots(figsize=(7, 4))
        if data.ndim == 1:
            ax.plot(time, data)
        else:
            for i in range(data.shape[1]):
                lbl = legend[i] if isinstance(legend, (list, tuple)) and i < len(legend) else (legend if isinstance(legend, str) else None)
                ax.plot(time, data[:, i], label=lbl)
        ax.set_title(title)
        if legend:
            ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel(dataName)
        ax.grid(True)
        self._finalize(fig, save, filename or self._slug_from_title(title, fallback="generic"), show, dpi)

    def plot_generic_data_subplot(self, data: np.ndarray, total_time: float, dt: float, title: str = "",
                                  legend: Sequence[str] | str = (), ylabel: str | Sequence[str] = "",
                                  xlabel: str = "Time", save: bool = True, filename: Optional[str] = None,
                                  show: bool = True, dpi: int = 150):
        assert data.ndim == 2, "data must be 2D for subplot plotting"
        time = self._time_vector(data.shape[0], dt, total_time)
        n_plots = data.shape[1]
        fig, axs = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots), sharex=True)
        fig.suptitle(title)
        for i in range(n_plots):
            ax = axs[i] if n_plots > 1 else axs
            lbl = legend[i] if isinstance(legend, (list, tuple)) and i < len(legend) else (legend if isinstance(legend, str) else None)
            ax.plot(time, data[:, i], label=lbl)
            if isinstance(ylabel, (list, tuple)) and i < len(ylabel):
                ax.set_ylabel(ylabel[i])
            elif isinstance(ylabel, str):
                ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            if lbl:
                ax.legend()
            ax.grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._finalize(fig, save, filename or self._slug_from_title(title, fallback="subplot"), show, dpi)

    def save_params(self, params: dict, filename: str = "params.json"):
        out = self.base_dir / filename
        with open(out, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

    # internals
    def _write_meta(self, meta: dict):
        out = self.base_dir / "meta.json"
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    def _time_vector(self, N: int, dt: float, total_time: Optional[float]):
        if total_time is None:
            return np.arange(N) * dt
        t = np.arange(0, total_time, dt)
        if len(t) != N:
            t = np.arange(N) * dt
        return t

    def _slug_from_title(self, title: str, fallback: str) -> str:
        return _slugify(title) if title else fallback

    def _fname(self, prefix: Optional[str], suffix: str) -> str:
        if prefix:
            return f"{_slugify(prefix)}_{suffix}"
        return suffix

    def _plot_single(self, data: np.ndarray, dt: float, total_time: Optional[float], ylabel: str, title: str, xlabel: str,
                     save: bool, filename: str, show: bool, dpi: int):
        time = self._time_vector(data.shape[0], dt, total_time)
        fig, ax = plt.subplots(figsize=(7, 4))
        if data.ndim == 1:
            ax.plot(time, data)
        else:
            for i in range(data.shape[1] if data.ndim == 2 else 1):
                y = data[:, i] if data.ndim == 2 else data
                ax.plot(time, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        self._finalize(fig, save, filename, show, dpi)

    def _finalize(self, fig, save: bool, filename_no_ext: str, show: bool, dpi: int):
        if save:
            fn = self.plots_dir / f"{filename_no_ext}.png"
            fig.savefig(fn, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)


def position_tracking_error(pos_actual, pos_desired):
    errors = np.linalg.norm(pos_actual - pos_desired, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    rmse = np.sqrt(np.mean(errors**2))
    return errors, mean_error, max_error, rmse
