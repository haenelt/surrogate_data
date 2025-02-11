# -*- coding: utf-8 -*-
"""Pytest library of surrogate_data."""

import numpy as np
import pytest

from surrogate_data.surrogate import Mask, distance_matrix


@pytest.fixture
def array() -> np.ndarray:
    """2D input array.."""
    return np.random.rand(128, 128)


def test_distance_matrix(array: np.ndarray) -> None:
    """Check size of distance matrix."""
    d = distance_matrix(array)
    assert np.shape(d) == (array.size, array.size)


def test_coordinates(array: np.ndarray) -> None:
    """Check if number of mask coordiantes match the masked pixels in the genrated
    mask."""
    m = Mask(array)
    roi = m.scatter
    assert np.shape(m.coordinates)[0] == np.count_nonzero(roi)
    assert np.shape(m.coordinates)[1] == 2
