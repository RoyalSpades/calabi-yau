from typing import List, Optional, Sequence
import numpy as np
import sympy as sp

from .ambient import WeightedProjectiveSpace

class CalabiYauHypersurface:
    """Hypersurface in a weighted projective space.

    CY condition for WP^n(w0..wn): weighted degree d equals sum(weights).
    """
    def __init__(self, ambient: WeightedProjectiveSpace, f: sp.Expr):
        self.ambient = ambient
        self.f = sp.expand(f)
        ok, deg = ambient.is_quasi_homogeneous(self.f)
        if not ok or deg is None:
            raise ValueError("f must be quasi-homogeneous in the ambient weights")
        self.degree = int(deg)
        self.grad = [sp.diff(self.f, z) for z in self.ambient.coords]

        # --- Numeric callables (faster + type-checker-friendly) ---
        # Use NumPy backend; all exponents are integers so complex eval is fine.
        self._f_num = sp.lambdify(self.ambient.coords, self.f, modules=["numpy"]) # -> complex scalar
        self._grad_num = sp.lambdify(self.ambient.coords, self.grad, modules=["numpy"]) # -> array-like of complex

    def is_calabi_yau(self) -> bool:
        return self.degree == sum(self.ambient.weights)

    def gradient_at_point(self, point: Sequence[complex]) -> np.ndarray:
        """Return ∇f evaluated at a projective chart point (as complex128 numpy array)."""
        vals = self._grad_num(*point) # list/ndarray of complex
        return np.asarray(vals, dtype=np.complex128).reshape(-1)

    def f_at_point(self, point: Sequence[complex]) -> complex:
        """Return f(point) as a built-in complex number."""
        val = self._f_num(*point)
        # Ensure a plain Python complex is returned (helps some callers / linters)
        return complex(np.asarray(val, dtype=np.complex128))

    def random_real_sample_on_chart(self, chart_idx: int = 0, n: int = 200,
                                    solve_idx: Optional[int] = None,
                                    seed: Optional[int] = 0) -> List[List[float]]:
        import random
        rng = random.Random(seed)
        z = list(self.ambient.coords)
        dimp1 = len(z)
        if chart_idx < 0 or chart_idx >= dimp1:
            raise ValueError("invalid chart index")
        cand = [i for i in range(dimp1) if i != chart_idx]
        if solve_idx is None:
            import random as _r
            solve_idx = rng.choice(cand)
        elif solve_idx == chart_idx:
            raise ValueError("solve_idx must differ from chart_idx")

        pts = []
        for _ in range(n):
            vals = [0.0] * dimp1
            vals[chart_idx] = 1.0
            for i in range(dimp1):
                if i in (chart_idx, solve_idx):
                    continue
                vals[i] = rng.uniform(-1.2, 1.2)
            xs = sp.symbols("x_solve")
            subs = {z[i]: (xs if i == solve_idx else vals[i]) for i in range(dimp1)}
            eq = sp.N(sp.expand(self.f.subs(subs)))
            root = None
            for guess in [rng.uniform(-1.2, 1.2), rng.uniform(-2.0, 2.0), 0.0]:
                try:
                    r = sp.nsolve(eq, xs, guess, tol=1e-14, maxsteps=200)
                    r = complex(r)
                    if abs(r.imag) < 1e-8:
                        root = r.real
                        break
                except Exception:
                    continue
            if root is None:
                continue
            vals[solve_idx] = float(root)
            pts.append(vals)
        return pts

    def smoothness_check_by_sampling(self, points: List[List[float]], eps: float = 1e-7) -> float:
        ok = 0
        for p in points:
            if abs(self.f_at_point(p)) > 1e-6:
                continue
            g = self.gradient_at_point(p)
            if float(np.linalg.norm(g)) > eps:
                ok += 1
        return ok / max(1, len(points))