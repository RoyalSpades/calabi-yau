from typing import Optional
import numpy as np
import math as _math
import plotly.graph_objects as go
from .slice import sample_quintic_slice

# Camera helper

def _camera_from_angles(yaw_deg: float = 45.0, pitch_deg: float = 30.0,
                        roll_deg: float = 0.0, R: float = 2.0) -> dict:
    yaw = _math.radians(yaw_deg)
    pitch = _math.radians(pitch_deg)
    roll = _math.radians(roll_deg)
    ex = R * _math.cos(pitch) * _math.cos(yaw)
    ey = R * _math.cos(pitch) * _math.sin(yaw)
    ez = R * _math.sin(pitch)
    eye_vec = np.array([ex, ey, ez], dtype=float)
    up0 = np.array([0.0, 0.0, 1.0], dtype=float)
    if np.linalg.norm(eye_vec) < 1e-12:
        up_vec = up0
    else:
        k = eye_vec / np.linalg.norm(eye_vec)
        v = up0
        v_par = np.dot(v, k) * k
        v_perp = v - v_par
        v_perp_rot = v_perp * _math.cos(roll) + np.cross(k, v_perp) * _math.sin(roll)
        up_vec = v_par + v_perp_rot
    return {"eye": {"x": float(ex), "y": float(ey), "z": float(ez)},
            "up":  {"x": float(up_vec[0]), "y": float(up_vec[1]), "z": float(up_vec[2])}}

# Plotly viewer with yaw/pitch/roll + play/pause + presets

def plotly_quintic_slice_with_controls(C: complex = 1+0j, n: int = 14000, r_max: float = 1.2,
                                       branches: str = "all", seed: int = 0,
                                       projector: Optional[np.ndarray] = None,
                                       point_size: float = 1.7, opacity: float = 0.6,
                                       radius: float = 2.0) -> None:
    Y = sample_quintic_slice(C=C, n=n, r_max=r_max, branches=branches, seed=seed, projector=projector)
    if Y.shape[0] == 0:
        raise RuntimeError("No points were sampled for the slice")
    fig = go.Figure(data=[go.Scatter3d(x=Y[:,0], y=Y[:,1], z=Y[:,2], mode='markers',
                                       marker=dict(size=point_size, opacity=opacity))])
    cam0 = _camera_from_angles(45, 30, 0, R=radius)
    fig.update_layout(scene=dict(aspectmode='data', camera=cam0),
                      margin=dict(l=0, r=0, t=35, b=0),
                      title="Quintic slice — interactive 3D (yaw/pitch/roll)")
    frames = []
    for a in range(0, 361, 5):
        cam = _camera_from_angles(a, 30, 0, R=radius)
        frames.append(go.Frame(name=f"yaw{a}", layout=dict(scene=dict(camera=cam))))
    fig.frames = frames
    slider_steps = [{"label": f"{a}°", "method": "animate",
                     "args": [[f"yaw{a}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]}
                    for a in range(0, 361, 15)]
    sliders = [{"active": 3, "currentvalue": {"prefix": "Yaw: ", "visible": True},
                "steps": slider_steps, "x": 0.10, "len": 0.80, "pad": {"t": 55}}]
    play_pause = {"type": "buttons", "direction": "left", "x": 0.10, "y": 1.12, "showactive": False,
                  "buttons": [
                      {"label": "Play", "method": "animate",
                       "args": [None, {"fromcurrent": True, "mode": "immediate", "frame": {"duration": 40, "redraw": False}, "transition": {"duration": 0}}]},
                      {"label": "Pause", "method": "animate",
                       "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]},
                  ]}
    presets = {"type": "dropdown", "x": 0.42, "y": 1.12, "showactive": True,
               "buttons": [
                   {"label": "Iso (45°,30°)", "method": "relayout", "args": [{"scene.camera": _camera_from_angles(45, 30, 0, R=radius)}]},
                   {"label": "Front (0°,0°)",  "method": "relayout", "args": [{"scene.camera": _camera_from_angles(0, 0, 0, R=radius)}]},
                   {"label": "Right (90°,0°)", "method": "relayout", "args": [{"scene.camera": _camera_from_angles(90, 0, 0, R=radius)}]},
                   {"label": "Top (0°,90°)",   "method": "relayout", "args": [{"scene.camera": _camera_from_angles(0, 90, 0, R=radius)}]},
               ]}
    pitch_menu = {"type": "dropdown", "x": 0.66, "y": 1.12, "showactive": True,
                  "buttons": [{"label": f"Pitch {p}°", "method": "relayout",
                               "args": [{"scene.camera": _camera_from_angles(45, p, 0, R=radius)}]} for p in [-60, -30, 0, 30, 60, 80]]}
    roll_menu = {"type": "dropdown", "x": 0.86, "y": 1.12, "showactive": True,
                 "buttons": [{"label": f"Roll {r}°", "method": "relayout",
                              "args": [{"scene.camera": _camera_from_angles(45, 30, r, R=radius)}]} for r in [0, 15, 30, 45, 60, 90]]}
    fig.update_layout(updatemenus=[play_pause, presets, pitch_menu, roll_menu], sliders=sliders)
    fig.show()