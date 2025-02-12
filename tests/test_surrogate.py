# -*- coding: utf-8 -*-
"""Pytest library of surrogate_data."""

import matplotlib.image as mpimg
import numpy as np
import pytest

from surrogate_data.surrogate import Mask, distance_matrix, smooth

# array dimensions
Nx = 128
Ny = 128


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


@pytest.fixture
def mask() -> np.ndarray:
    """2D binary mask."""
    image = np.zeros((Nx, Ny))
    image[40:60, 40:60] = 1
    return image


def test_distance_matrix(array: np.ndarray) -> None:
    """Check size of distance matrix."""
    d = distance_matrix(array)
    assert np.shape(d) == (array.size, array.size)


def test_coordinates(array: np.ndarray) -> None:
    """Check if number of mask coordiantes match the masked pixels in the genrated
    mask."""
    m = Mask(array)
    roi = m.scatter
    assert m.coordinates is not None, "m.coordinates is None"
    assert np.shape(m.coordinates)[0] == np.count_nonzero(roi)
    assert np.shape(m.coordinates)[1] == 2


def test_smooth(image: np.ndarray, mask: np.ndarray) -> None:
    """Check if untouched pixels are really untouched when using a mask during
    smoothing."""
    k = 100
    image_smoothed = smooth(image, k)
    assert image_smoothed[0, 0] != image[0, 0]
    assert image_smoothed[50, 50] != image[50, 50]
    image_smoothed = smooth(image, k, mask)
    assert image_smoothed[0, 0] != image[0, 0]
    assert image_smoothed[50, 50] == image[50, 50]
