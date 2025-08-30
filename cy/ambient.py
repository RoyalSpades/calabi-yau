from dataclasses import dataclass
from typing import Tuple, Any
import sympy as sp


@dataclass
class WeightedProjectiveSpace:
    dim: int
    weights: Tuple[int, ...]


    def __post_init__(self):
        if len(self.weights) != self.dim + 1:
            raise ValueError("weights must have length dim+1")
        if any(w <= 0 for w in self.weights):
            raise ValueError("weights must be positive integers")
        self.coords = sp.symbols("z0:" + str(self.dim + 1))


    def weighted_degree(self, term: Any) -> int:
        if isinstance(term, (list, tuple)):
            term = sum(sp.sympify(t) for t in term)
        expr = sp.sympify(term)
        if not isinstance(expr, sp.Expr):
            raise TypeError("weighted_degree expects a SymPy Expr-like term")
        expr = sp.expand(expr)
        if expr.is_Add:
            raise ValueError("weighted_degree expects a single term (no '+')")
        deg = 0
        for i, z in enumerate(self.coords):
            exp = sp.degree(expr, z)
            if exp is None or exp is sp.S.NegativeInfinity:
                exp = 0
            deg += int(self.weights[i]) * int(exp)
        return int(deg)


    def is_quasi_homogeneous(self, f: sp.Expr) -> Tuple[bool, int]:
        f = sp.expand(f)
        terms = f.as_ordered_terms()
        if not terms:
            return False, 0
        wd = self.weighted_degree(terms[0])
        for t in terms[1:]:
            if self.weighted_degree(t) != wd:
                return False, 0
        return True, int(wd)