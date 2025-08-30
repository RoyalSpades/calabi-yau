from typing import Tuple, Optional, Iterable
import numpy as np
import math
from .slice import FIFTH_ROOTS_UNITY, principal_nth_root

# Grid triangulation over (r,theta)

def _mesh_grid_indices(nr: int, nt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    faces_i, faces_j, faces_k = [], [], []
    for r in range(nr - 1):
        for t in range(nt):
            t_next = (t + 1) % nt
            a = r * nt + t
            b = (r + 1) * nt + t
            c = (r + 1) * nt + t_next
            d = r * nt + t_next
            faces_i.extend([a, a])
            faces_j.extend([b, c])
            faces_k.extend([c, d])
    return np.array(faces_i, dtype=int), np.array(faces_j, dtype=int), np.array(faces_k, dtype=int)

# Build one branch surface

def surface_grid_quintic_slice(
    C: complex = 1 + 0j,
    r_max: float = 1.2,
    nr: int = 120,
    ntheta: int = 240,
    branch_index: int = 0,
    projector: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if projector is None:
        projector = np.array([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0]], dtype=float)
    projector = np.asarray(projector, dtype=float)
    if projector.shape != (3, 4):
        raise ValueError("projector must be 3×4")

    rs = np.linspace(1e-6, r_max, nr)
    thetas = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)

    XX, YY, ZZ = [], [], []
    root_phase = FIFTH_ROOTS_UNITY[branch_index % 5]

    for r in rs:
        for th in thetas:
            z1 = r * (np.cos(th) + 1j * np.sin(th))
            w = C - z1 ** 5
            z2p = principal_nth_root(w, 5)
            z2 = z2p * root_phase
            v4 = np.array([z1.real, z1.imag, z2.real, z2.imag], dtype=float)
            v3 = projector @ v4
            XX.append(v3[0]); YY.append(v3[1]); ZZ.append(v3[2])

    X = np.array(XX, dtype=float)
    Y = np.array(YY, dtype=float)
    Z = np.array(ZZ, dtype=float)
    i, j, k = _mesh_grid_indices(nr, ntheta)
    return X, Y, Z, i, j, k

# Plot surfaces for one or more branches

def plotly_quintic_slice_surface(
    C: complex = 1 + 0j,
    r_max: float = 1.2,
    nr: int = 120,
    ntheta: int = 240,
    branches: str | Iterable[int] = "principal",
    projector: Optional[np.ndarray] = None,
    opacity: float = 0.5,
    color: str | None = None,
    lighting: Optional[dict] = None,
    title: str = "Quintic slice — semi-transparent surface",
) -> None:
    import plotly.graph_objects as go

    if lighting is None:
        lighting = dict(ambient=0.4, diffuse=0.7, specular=0.2, roughness=0.9, fresnel=0.2)

    if branches == "principal":
        branch_list = [0]
    elif branches == "all":
        branch_list = [0,1,2,3,4]
    else:
        branch_list = list(branches)

    traces = []
    for bi in branch_list:
        X, Y, Z, i, j, k = surface_grid_quintic_slice(C=C, r_max=r_max, nr=nr, ntheta=ntheta,
                                                      branch_index=bi, projector=projector)
        traces.append(go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=k,
                                name=f"branch {bi}", opacity=opacity,
                                color=color, lighting=lighting, flatshading=False))

    fig = go.Figure(data=traces)
    fig.update_layout(scene=dict(aspectmode='data'), title=title,
                      margin=dict(l=0, r=0, t=35, b=0))
    fig.show()