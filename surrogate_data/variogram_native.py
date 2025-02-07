# -*- coding: utf-8 -*-
"""Utility functions."""

from functools import lru_cache

import numpy as np

__all__ = ["Variogram"]


class Variogram:
    # cythonize
    # faster?
    # docstrings
    # test

    def __init__(
        self,
        data: np.ndarray,
        mask: np.ndarray | None = None,
        delta: float = 1.0,
        n_dist: int = 25,
    ) -> None:
        self.data = data
        self.mask = np.zeros_like(data) if mask is None else mask
        self.delta = delta
        self.n_dist = n_dist

        self.bw = 6 * delta
        self.dist_points = np.linspace(0, 2 * delta * n_dist, n_dist)
        self.nx, self.ny = np.shape(self.data)
        self.n = self.nx * self.ny

    @property
    @lru_cache()
    def data_flatten(self) -> np.ndarray:
        return self.data.flatten()

    @property
    @lru_cache()
    def mask_flatten(self) -> np.ndarray:
        return self.mask.flatten()

    @property
    @lru_cache()
    def coord_flat(self) -> np.ndarray:
        x, y = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        x_flat = x.flatten()
        y_flat = y.flatten()
        return np.column_stack((x_flat, y_flat))

    def run(self) -> np.ndarray:
        _var = np.zeros(self.n_dist)
        for i, _d in enumerate(self.dist_points):
            print(i)
            _var[i] = self.gamma(_d)
        return _var

    def kernel(self, dist: float) -> float:
        return np.exp(-((2.68 * dist) ** 2) / (2 * self.bw) ** 2)

    def gamma(self, dist_point: float) -> float:
        numerator = 0.0
        denominator = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.mask_flatten[i] != 0:
                    continue
                val_i = self.data_flatten[i]
                val_j = self.data_flatten[j]
                _dist = self.distance(dist_point, i, j)
                numerator += self.kernel(_dist) * self.var(val_i, val_j)
                denominator += self.kernel(_dist)

        return numerator / denominator

    def distance(self, dist_point: float, idx1: int, idx2: int) -> float:
        pt1 = self.coord_flat[idx1, :]
        pt2 = self.coord_flat[idx2, :]
        _dist = np.linalg.norm(pt1 - pt2)
        return np.abs(dist_point - _dist)

    @staticmethod
    def var(val1: float, val2: float) -> float:
        return 0.5 * (val1 - val2) ** 2
