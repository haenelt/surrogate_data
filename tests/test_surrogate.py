# -*- coding: utf-8 -*-
"""Pytest library of surrogate_data."""

import matplotlib.image as mpimg
import numpy as np
import pytest

from surrogate_data.surrogate import SurrogateMap, distance_matrix

# array dimensions
Nx = 128  # height of test image
Ny = 128  # width of test image
ATOL = 1e-7  # absolute tolerance for the comparison of floats


@pytest.fixture
def array() -> np.ndarray:
    """2D input array.."""
    return np.random.rand(Nx, Ny)


@pytest.fixture
def image() -> np.ndarray:
    """Test image."""
    file_img = "./tests/data/lenna.png"
    image = mpimg.imread(file_img)
    image = np.mean(image, axis=2)
    return image.astype(np.float64)[:Nx, :Ny]


def test_distance_matrix(array: np.ndarray) -> None:
    """Check size of distance matrix."""
    d = distance_matrix(array)
    assert np.shape(d) == (array.size, array.size)


def test_smooth(image: np.ndarray) -> None:
    """Check if smoothing changes the input array."""
    k = 100
    surr = SurrogateMap(image)
    image_smoothed = surr._smooth(image, k)
    assert image_smoothed[0, 0] != image[0, 0]
    assert image_smoothed[50, 50] != image[50, 50]


def test_scale(array: np.ndarray) -> None:
    """Check if scaling matches min-max of references array."""
    height, width = np.shape(array)
    array_0 = np.random.normal(0, 10, size=(height, width))
    _min = np.min(array_0)
    _max = np.max(array_0)
    surr = SurrogateMap(array_0)
    array_scaled = surr._scale(array)
    assert np.isclose(np.min(array_scaled), _min, atol=ATOL)
    assert np.isclose(np.max(array_scaled), _max, atol=ATOL)


def test_smooth_scale(image: np.ndarray) -> None:
    """Check if smoothing changes the input array when scaling performed after."""
    k = 100
    surr = SurrogateMap(image)
    image_smoothed = surr._smooth(image, k)
    image_scaled = surr._scale(image_smoothed)
    assert image_scaled[0, 0] != image[0, 0]
    assert image_scaled[50, 50] != image[50, 50]


def test_parameters(array: np.ndarray) -> None:
    """Check kwargs arguments."""
    surr = SurrogateMap(array)
    assert surr.num_cores == 4
    surr = SurrogateMap(array, num_cores=8)
    assert surr.num_cores == 8
    surr = SurrogateMap(array, num_cores=12, bw=2.0)
    assert surr.num_cores == 12
    assert np.isclose(surr.bw, 2.0, atol=ATOL)
