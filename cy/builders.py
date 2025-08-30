from dataclasses import dataclass
from typing import Tuple
import sympy as sp
from .ambient import WeightedProjectiveSpace
from .hypersurface import CalabiYauHypersurface

@dataclass
class KnownInvariants:
    h11: int
    h21: int
    euler: int

def fermat_quintic() -> CalabiYauHypersurface:
    ambient = WeightedProjectiveSpace(dim=4, weights=(1,1,1,1,1))
    z0, z1, z2, z3, z4 = ambient.coords
    f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5
    X = CalabiYauHypersurface(ambient, f)
    assert X.is_calabi_yau()
    return X

def random_quintic(coeff_range: Tuple[int,int] = (-2,2), seed: int = 0) -> CalabiYauHypersurface:
    import random
    rng = random.Random(seed)
    ambient = WeightedProjectiveSpace(dim=4, weights=(1,1,1,1,1))
    z = ambient.coords
    monoms = []
    for a0 in range(6):
        for a1 in range(6 - a0):
            for a2 in range(6 - a0 - a1):
                for a3 in range(6 - a0 - a1 - a2):
                    a4 = 5 - (a0 + a1 + a2 + a3)
                    monoms.append((z[0]**a0) * (z[1]**a1) * (z[2]**a2) * (z[3]**a3) * (z[4]**a4))
    coeffs = [rng.randint(coeff_range[0], coeff_range[1]) for _ in monoms]
    if all(c == 0 for c in coeffs):
        coeffs[rng.randrange(len(coeffs))] = 1
    f = sum(c*m for c,m in zip(coeffs, monoms))
    return CalabiYauHypersurface(ambient, f)

def invariants_quintic() -> KnownInvariants:
    h11, h21 = 1, 101
    return KnownInvariants(h11=h11, h21=h21, euler=2*(h11 - h21))