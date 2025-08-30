from typing import Optional
import numpy as np
import plotly.graph_objects as go
from .slice import sample_quintic_slice

# Marching-cubes iso-surface from a point cloud

def plotly_quintic_slice_iso_surface(
    C: complex = 1+0j,
    branches: str = "all",
    n_points: int = 40000,
    r_max: float = 1.2,
    projector: Optional[np.ndarray] = None,
    grid_n: int = 72,
    margin: float = 0.06,
    iso: Optional[float] = None,
    seed: int = 0,
    opacity: float = 0.45,
    color: Optional[str] = None,
    title: str = "Quintic slice — marching-cubes iso-surface",
) -> None:
# --- SciPy imports with safe fallback (avoids Pylance unknown symbol warning) ---
    try:
        from scipy.spatial import cKDTree as _KDTree # type: ignore # fast C implementation
        _USE_CKD = True
    except Exception:
        from scipy.spatial import KDTree as _KDTree # pure Python fallback
        _USE_CKD = False
    try:
        from skimage.measure import marching_cubes
    except Exception as e:
        raise RuntimeError("scikit-image is required: pip install scikit-image") from e

    if projector is None:
        projector = np.array([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0]], dtype=float)

    # 1) Dense point cloud
    Y = sample_quintic_slice(C=C, n=n_points, r_max=r_max, branches=branches, seed=seed, projector=projector)
    if Y.shape[0] == 0:
        raise RuntimeError("Sampling produced no points; adjust parameters.")

    # 2) grid bounds
    mins = Y.min(axis=0); maxs = Y.max(axis=0)
    span = maxs - mins
    mins = mins - margin * span
    maxs = maxs + margin * span

    nx = ny = nz = grid_n
    xs = np.linspace(mins[0], maxs[0], nx)
    ys = np.linspace(mins[1], maxs[1], ny)
    zs = np.linspace(mins[2], maxs[2], nz)
    dx = (maxs - mins) / np.array([nx-1, ny-1, nz-1])

    # 3) Distance field via KD-tree
    tree = _KDTree(Y)
    vol = np.empty((nx, ny, nz), dtype=float)
    for k in range(nz):
        Zk = zs[k]
        Xg, Yg = np.meshgrid(xs, ys, indexing='ij')
        Q = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, Zk)])
        try:
            if _USE_CKD:
                d, _ = tree.query(Q, k=1, workers=-1)
            else:
                d, _ = tree.query(Q, k=1)
        except TypeError:
            # Some SciPy builds may not support 'workers' even for cKDTree
            d, _ = tree.query(Q, k=1)
        vol[:, :, k] = d.reshape((nx, ny))

    # 4) Marching cubes at iso threshold
    if iso is None:
        voxel_diag = float(np.linalg.norm(dx))
        iso = 1.5 * voxel_diag

    verts, faces, normals, values = marching_cubes(vol, level=iso,
                                                   spacing=(xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]))
    
    # Convert voxel-space verts to world coords by offsetting by mins
    verts_world = verts + np.array([mins[0], mins[1], mins[2]])

    i = faces[:,0].astype(int)
    j = faces[:,1].astype(int)
    k = faces[:,2].astype(int)

    mesh = go.Mesh3d(x=verts_world[:,0], y=verts_world[:,1], z=verts_world[:,2],
                     i=i, j=j, k=k, opacity=opacity, color=color,
                     name="iso-surface", flatshading=False)

    fig = go.Figure(data=[mesh])
    fig.update_layout(scene=dict(aspectmode='data'),
                      margin=dict(l=0, r=0, t=40, b=0),
                      title=title + f" (iso={iso:.3g}, grid={grid_n}³, points={Y.shape[0]})")
    fig.show()