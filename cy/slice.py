from typing import Optional, List
import numpy as np
import cmath, math
import sympy as sp

# Roots of unity for branch selection
FIFTH_ROOTS_UNITY = [cmath.exp(2j*math.pi*k/5.0) for k in range(5)]

# Affine slice constant C from fixed a=z3/z5, b=z4/z5

def c_from_affine_constants(a: complex, b: complex) -> complex:
    return - (1 + a**5 + b**5)

# n-th roots helpers

def principal_nth_root(w: complex, n: int) -> complex:
    if w == 0:
        return 0j
    r, theta = cmath.polar(w)
    return (r ** (1.0/n)) * cmath.exp(1j * theta / n)

def all_nth_roots(w: complex, n: int) -> List[complex]:
    base = principal_nth_root(w, n)
    return [base * cmath.exp(2j*math.pi*k/n) for k in range(n)]

# Point sampler and 3D projections

def sample_quintic_slice(C: complex = 1+0j, n: int = 20000, r_max: float = 1.2,
                         branches: str = "all", seed: int = 0,
                         projector: Optional[np.ndarray] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if projector is None:
        projector = np.array([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0]], dtype=float)
    projector = np.asarray(projector, dtype=float)
    if projector.shape != (3,4):
        raise ValueError("projector must be 3×4")

    pts = []
    count = 0
    while count < n:
        u, v = rng.random(), rng.random()
        r = r_max * math.sqrt(u)
        theta = 2*math.pi*v
        z1 = r * math.cos(theta) + 1j * r * math.sin(theta)
        w = C - z1**5
        roots = [principal_nth_root(w, 5)] if branches == 'principal' else all_nth_roots(w, 5)
        for z2 in roots:
            x = np.array([z1.real, z1.imag, z2.real, z2.imag], dtype=float)
            y = projector @ x
            pts.append(y)
        count += 1
    return np.vstack(pts) if pts else np.zeros((0,3), dtype=float)

# Basic visualization helpers

def visualize_quintic_slice(C: complex = 1+0j, n: int = 8000, r_max: float = 1.2,
                             branches: str = "all", seed: int = 0,
                             projector: Optional[np.ndarray] = None) -> None:
    import matplotlib.pyplot as plt
    Y = sample_quintic_slice(C=C, n=n, r_max=r_max, branches=branches, seed=seed, projector=projector)
    if Y.shape[0] == 0:
        raise RuntimeError("No points were sampled for the slice")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Y[:,0], Y[:,1], Y[:,2], s=0.5, alpha=0.6)
    ax.set_title("2D slice of the quintic: z1^5 + z2^5 = C (projected)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    plt.show()


def plotly_quintic_slice(C: complex = 1+0j, n: int = 12000, r_max: float = 1.2,
                         branches: str = "all", seed: int = 0,
                         projector: Optional[np.ndarray] = None,
                         point_size: float = 1.8, opacity: float = 0.6) -> None:
    import plotly.graph_objects as go
    Y = sample_quintic_slice(C=C, n=n, r_max=r_max, branches=branches, seed=seed, projector=projector)
    if Y.shape[0] == 0:
        raise RuntimeError("No points were sampled for the slice")
    fig = go.Figure(data=[go.Scatter3d(x=Y[:,0], y=Y[:,1], z=Y[:,2], mode='markers',
                                       marker=dict(size=point_size, opacity=opacity))])
    fig.update_layout(scene=dict(aspectmode='data'), title="Quintic slice — points (Plotly)")
    fig.show()