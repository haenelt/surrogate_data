# -*- coding: utf-8 -*-
# distutils: language=c++
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
"""Utility functions."""

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, pow, abs

__all__ = ["Variogram"]


cdef class Variogram:
    # cythonize
    # faster?
    # docstrings
    # test
    # add mask?

    cdef double[:,:] data
    cdef double delta
    cdef int n_dist
    cdef double bw
    cdef double[:] dist_points
    cdef int nx
    cdef int ny
    cdef int n
    cdef float[:] data_flatten
    cdef double[:, :] coord_flat

    def __init__(
        self,
        data: np.ndarray,
        delta: float = 1.0,
        n_dist: int = 25,
    ) -> None:
        self.data = data
        self.delta = delta
        self.n_dist = n_dist

        self.bw = 6 * delta
        self.dist_points = np.linspace(0, 2 * delta * n_dist, n_dist)
        self.nx, self.ny = np.shape(self.data)
        self.n = self.nx * self.ny

        x, y = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        x_flat = x.ravel().astype(float)
        y_flat = y.ravel().astype(float)
        self.coord_flat = np.column_stack((x_flat, y_flat))

    def run(self):
        cdef double[:] _var = np.zeros(self.n_dist, dtype=np.float64)
        cdef np.ndarray[double, ndim=2] _data = np.array(self.data)
        cdef int i
        cdef np.ndarray[double, ndim=1] data_flatten = flatten_2d_array(_data)
        for i in range(len(self.dist_points)):
            print(i)
            pt1_x = self.coord_flat[:, 0]
            pt1_y = self.coord_flat[:, 1]
            pt2_x = self.coord_flat[:, 0]
            pt2_y = self.coord_flat[:, 1]
            _var[i] = gamma(data_flatten, self.dist_points[i], pt1_x, pt1_y, pt2_x, pt2_y, self.bw)
        return _var


cdef double gamma(np.ndarray[double,ndim=1] data, double dist_point, double[:] pt1_x, double[:] pt1_y, double[:] pt2_x, double[:] pt2_y, double bw):
    cdef float numerator = 0
    cdef float denominator = 0
    cdef double _dist
    cdef double data_diff
    cdef int i, j
    cdef double kernel_value

    cdef double factor = 2.68 * _dist
    cdef double exp_argument = - factor * factor / pow(2 * bw, 2)
    cdef int n = len(data)

    for i in range(n):
        for j in range(i + 1, n):
            _dist = abs(dist_point - sqrt(pow(pt1_x[i] - pt2_x[j], 2) + pow(pt1_y[i] - pt2_y[j], 2)))
            factor = 2.68 * _dist
            exp_argument = - factor * factor / pow(2 * bw, 2)
            kernel_value = exp(exp_argument)
            data_diff = data[i] - data[j]
            numerator += kernel_value * 0.5 * data_diff * data_diff
            denominator += kernel_value

    return numerator / denominator

# Cython function to flatten a 2D NumPy array
cdef flatten_2d_array(np.ndarray[np.float64_t, ndim=2] arr):
    cdef int i, j
    cdef int rows = arr.shape[0]
    cdef int cols = arr.shape[1]
    cdef np.ndarray[np.float64_t, ndim=1] flattened = np.empty(rows * cols, dtype=np.float64)

    cdef int index = 0
    for i in range(rows):
        for j in range(cols):
            flattened[index] = arr[i, j]
            index += 1

    return flattened
