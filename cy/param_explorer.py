from typing import Optional, Iterable, List, Tuple
import numpy as np
import plotly.graph_objects as go
from .slice import sample_quintic_slice
from .mesh import surface_grid_quintic_slice

# Points explorer (animate phase of C)

def plotly_quintic_param_explorer_points(
    C_mag: float = 1.0,
    phase_steps: int = 16,
    n_per_phase: int = 6000,
    r_max: float = 1.2,
    branches: str = "all",
    seed: int = 0,
    projector: Optional[np.ndarray] = None,
    point_size: float = 1.6,
    opacity: float = 0.65,
    title: str = "Quintic slice — parameter explorer (points)",
) -> None:
    if projector is None:
        projector = np.array([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0]], dtype=float)
    phases = np.linspace(0.0, 2*np.pi, phase_steps, endpoint=False)
    frames = []
    first = None
    for idx, ph in enumerate(phases):
        C = C_mag * np.exp(1j * ph)
        Y = sample_quintic_slice(C=C, n=n_per_phase, r_max=r_max,
                                 branches=branches, seed=seed+idx, projector=projector)
        if Y.shape[0] == 0:
            continue
        frames.append(go.Frame(name=f"phase{idx}", data=[go.Scatter3d(x=Y[:,0], y=Y[:,1], z=Y[:,2], mode='markers',
                                                                       marker=dict(size=point_size, opacity=opacity))],
                               layout=dict(title=go.layout.Title(text=f"{title} — phase={np.degrees(ph):.1f}°"))) )
        if first is None:
            first = Y
    if first is None:
        raise RuntimeError("No frames generated; adjust params.")
    fig = go.Figure(data=[go.Scatter3d(x=first[:,0], y=first[:,1], z=first[:,2], mode='markers',
                                       marker=dict(size=point_size, opacity=opacity))])
    fig.update_layout(scene=dict(aspectmode='data'), title=title, frames=frames,
                      margin=dict(l=0, r=0, t=40, b=0))
    slider_steps = [{"label": f"{int(360*i/phase_steps)}°", "method": "animate",
                     "args": [[f"phase{i}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                    for i in range(len(frames))]
    sliders = [{"active": 0, "currentvalue": {"prefix": "Phase: ", "visible": True},
                "steps": slider_steps, "x": 0.1, "len": 0.8, "pad": {"t": 50}}]
    play_pause = {"type": "buttons", "direction": "left", "x": 0.1, "y": 1.12, "showactive": False,
                  "buttons": [
                      {"label": "Play", "method": "animate",
                       "args": [None, {"fromcurrent": True, "mode": "immediate", "frame": {"duration": 120, "redraw": True}, "transition": {"duration": 0}}]},
                      {"label": "Pause", "method": "animate",
                       "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]},
                  ]}
    fig.update_layout(updatemenus=[play_pause], sliders=sliders)
    fig.show()

# Utilities for surface explorer

def _branches_to_list(branches: str | Iterable[int]) -> List[int]:
    if branches == "principal":
        return [0]
    if branches == "all":
        return [0,1,2,3,4]
    return list(branches)


def _phase_meshes_for_branches(C: complex, branches_list: List[int], r_max: float,
                               nr: int, ntheta: int, projector: Optional[np.ndarray]):
    meshes = []
    for bi in branches_list:
        X, Y, Z, i, j, k = surface_grid_quintic_slice(C=C, r_max=r_max, nr=nr, ntheta=ntheta,
                                                      branch_index=bi, projector=projector)
        meshes.append((X, Y, Z, i, j, k))
    return meshes

# Surface explorer (animate meshes as phase varies)

def plotly_quintic_param_explorer_surface(
    C_mag: float = 1.0,
    phase_steps: int = 12,
    r_max: float = 1.2,
    nr: int = 80,
    ntheta: int = 160,
    branches: str | Iterable[int] = "principal",
    projector: Optional[np.ndarray] = None,
    opacity: float = 0.5,
    color: str | None = None,
    lighting: Optional[dict] = None,
    title: str = "Quintic slice — surface parameter explorer",
) -> None:
    if projector is None:
        projector = np.array([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0]], dtype=float)
    if lighting is None:
        lighting = dict(ambient=0.4, diffuse=0.7, specular=0.25, roughness=0.9, fresnel=0.15)
    branches_list = _branches_to_list(branches)
    phases = np.linspace(0.0, 2*np.pi, phase_steps, endpoint=False)

    # Seed figure with first phase
    C0 = C_mag * np.exp(1j * phases[0])
    first_meshes = _phase_meshes_for_branches(C0, branches_list, r_max, nr, ntheta, projector)
    traces = []
    for idx, (X, Y, Z, i, j, k) in enumerate(first_meshes):
        traces.append(go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=k, opacity=opacity,
                                color=color, lighting=lighting, name=f"branch {branches_list[idx]}"))
    fig = go.Figure(data=traces)
    fig.update_layout(scene=dict(aspectmode='data'),
                      margin=dict(l=0, r=0, t=40, b=0),
                      title=title + " — phase=0°")

    # Build frames
    frames = []
    for p_idx, ph in enumerate(phases):
        C = C_mag * np.exp(1j * ph)
        meshes = _phase_meshes_for_branches(C, branches_list, r_max, nr, ntheta, projector)
        frame_traces = []
        for (X, Y, Z, i, j, k) in meshes:
            frame_traces.append(go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=k,
                                          opacity=opacity, color=color, lighting=lighting))
        frames.append(go.Frame(name=f"phase{p_idx}", data=frame_traces,
                               layout=dict(title=go.layout.Title(text=f"{title} — phase={int(np.degrees(ph))}°"))))
    fig.frames = frames

    slider_steps = [{"label": f"{int(360*i/phase_steps)}°", "method": "animate",
                     "args": [[f"phase{i}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                    for i in range(len(frames))]
    sliders = [{"active": 0, "currentvalue": {"prefix": "Phase: ", "visible": True},
                "steps": slider_steps, "x": 0.1, "len": 0.8, "pad": {"t": 50}}]
    play_pause = {"type": "buttons", "direction": "left", "x": 0.1, "y": 1.12, "showactive": False,
                  "buttons": [
                      {"label": "Play", "method": "animate",
                       "args": [None, {"fromcurrent": True, "mode": "immediate", "frame": {"duration": 160, "redraw": True}, "transition": {"duration": 0}}]},
                      {"label": "Pause", "method": "animate",
                       "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]},
                  ]}
    fig.update_layout(updatemenus=[play_pause], sliders=sliders)
    fig.show()