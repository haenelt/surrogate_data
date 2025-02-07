# -*- coding: utf-8 -*-
"""Pytest library of surrogate_data."""

import numpy as np

from surrogate_data.gauss import RandomGaussianField


def test_spat_freq() -> None:
    """Test minimum of spatial frequency magnitude vector."""
    gauss = RandomGaussianField(n=100)
    k = gauss.spat_freq
    assert np.isclose(np.min(k), 1e-10)


def test_noise() -> None:
    """Test statistics of white noise process."""
    gauss = RandomGaussianField(n=100)
    noise = gauss.noise(0, 1)
    assert np.isclose(np.mean(noise.real), 0.0, atol=0.1)
    assert np.isclose(np.std(noise.real), 1.0, atol=0.1)
    assert np.isclose(np.mean(noise.imag), 0.0, atol=0.1)
    assert np.isclose(np.std(noise.imag), 1.0, atol=0.1)


def test_sample() -> None:
    """Test statistics of resulting random gaussian field."""
    gauss = RandomGaussianField(n=100)
    sample = gauss.sample()
    assert np.isclose(np.mean(sample), 0.0)
    assert np.isclose(np.std(sample), 1.0)
