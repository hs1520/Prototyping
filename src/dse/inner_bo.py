"""Inner-layer Bayesian optimization — bilevel DSE continuous parameter search.

Given a fixed architecture (chosen by the outer MO-MCTS), the inner layer tunes
continuous parameters (e.g. control_frequency_hz). This is the "continuous" half
of the heterogeneous bilevel design: outer = discrete structure via MCTS, inner =
continuous parameters via Bayesian optimization (GP surrogate + Expected
Improvement). See docs/DSE_REDESIGN.md §五.

Pure-Python (no numpy/scipy): a 1-D GP with an RBF kernel solved by Cholesky, and
EI maximised over a candidate grid. BO is sample-efficient by design, so the
matrices stay tiny (n_init + n_iter points). Extensible to D dimensions later.
"""
from __future__ import annotations

import math
from statistics import NormalDist
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

Vector = List[float]
Matrix = List[List[float]]


# ---------------------------------------------------------------------------
# Minimal linear algebra (Cholesky solve)
# ---------------------------------------------------------------------------


def _cholesky(a: Matrix) -> Matrix:
    """Lower-triangular Cholesky factor of a symmetric positive-definite matrix."""
    n = len(a)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(a[i][i] - s, 1e-12))
            else:
                L[i][j] = (a[i][j] - s) / L[j][j]
    return L


def _solve_lower(L: Matrix, b: Vector) -> Vector:
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    return y


def _solve_upper(LT: Matrix, b: Vector) -> Vector:
    # LT is L transposed (upper); solve LT x = b by back-substitution
    n = len(LT)
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (b[i] - sum(LT[i][k] * x[k] for k in range(i + 1, n))) / LT[i][i]
    return x


def _chol_solve(L: Matrix, b: Vector) -> Vector:
    """Solve A x = b given A = L Lᵀ."""
    y = _solve_lower(L, b)
    n = len(L)
    LT = [[L[j][i] for j in range(n)] for i in range(n)]
    return _solve_upper(LT, y)


# ---------------------------------------------------------------------------
# Standard normal (for Expected Improvement) — stdlib, still numpy/scipy-free
# ---------------------------------------------------------------------------


_STD_NORMAL = NormalDist()


# ---------------------------------------------------------------------------
# Bayesian optimizer (1-D, maximisation)
# ---------------------------------------------------------------------------


@dataclass
class BOResult:
    best_x: float
    best_y: float
    n_evals: int


class BayesianOptimizer:
    """1-D Bayesian optimization with a GP (RBF kernel) and EI acquisition."""

    def __init__(
        self,
        bounds: Tuple[float, float],
        objective: Callable[[float], float],
        n_init: int = 3,
        n_iter: int = 10,
        length_scale: Optional[float] = None,
        signal_var: float = 1.0,
        noise: float = 1e-4,
        xi: float = 0.01,
        grid: int = 200,
        random_seed: Optional[int] = None,
    ):
        self.lo, self.hi = bounds
        self.objective = objective
        self.n_init = n_init
        self.n_iter = n_iter
        self.length_scale = length_scale or (self.hi - self.lo) / 5.0
        self.signal_var = signal_var
        self.noise = noise
        self.xi = xi
        self.grid = grid
        self.rng = random.Random(random_seed)
        self._xs: Vector = []
        self._ys: Vector = []

    def _kernel(self, a: float, b: float) -> float:
        d = (a - b) / self.length_scale
        return self.signal_var * math.exp(-0.5 * d * d)

    def _posterior(self, xstar: float, L: Matrix, alpha: Vector) -> Tuple[float, float]:
        ks = [self._kernel(xstar, xi) for xi in self._xs]
        mean = sum(k * a for k, a in zip(ks, alpha))
        v = _solve_lower(L, ks)
        var = self._kernel(xstar, xstar) - sum(vi * vi for vi in v)
        return mean, max(var, 1e-12)

    def _ei(self, mean: float, var: float, best: float) -> float:
        sigma = math.sqrt(var)
        if sigma < 1e-9:
            return 0.0
        improvement = mean - best - self.xi
        z = improvement / sigma
        return improvement * _STD_NORMAL.cdf(z) + sigma * _STD_NORMAL.pdf(z)

    def optimize(self) -> BOResult:
        # initial design
        for _ in range(self.n_init):
            x = self.rng.uniform(self.lo, self.hi)
            self._xs.append(x)
            self._ys.append(self.objective(x))

        for _ in range(self.n_iter):
            n = len(self._xs)
            K = [
                [self._kernel(self._xs[i], self._xs[j]) + (self.noise if i == j else 0.0)
                 for j in range(n)]
                for i in range(n)
            ]
            L = _cholesky(K)
            alpha = _chol_solve(L, self._ys)
            best = max(self._ys)

            # maximise EI over a candidate grid
            best_ei, next_x = -1.0, self.lo
            for g in range(self.grid + 1):
                cand = self.lo + (self.hi - self.lo) * g / self.grid
                mean, var = self._posterior(cand, L, alpha)
                ei = self._ei(mean, var, best)
                if ei > best_ei:
                    best_ei, next_x = ei, cand

            self._xs.append(next_x)
            self._ys.append(self.objective(next_x))

        bi = max(range(len(self._ys)), key=lambda i: self._ys[i])
        return BOResult(best_x=self._xs[bi], best_y=self._ys[bi], n_evals=len(self._ys))
