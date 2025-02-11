# -*- coding: utf-8 -*-
"""Compute null distribution."""

import numpy as np
from scipy.spatial.distance import pdist, squareform

__all__ = ["variogram", "distance_matrix", "permute", "Mask"]


def variogram(array: np.ndarray, lag: float, bw: float) -> float:
    """Computes the variogram gamma(h) using the element positions of a 2D array as
    spatial coordinates. This function computes the smoothed variogram as outlined in
    [1, 2].

    Args:
        array: 2D array of values.
        lag: Lag distance.
        bw: Bandwidth parameter controlling smoothness.

    Returns:
        Estimated variogram value gamma(h).

    References:
    .. [1] Viladomat et al., Assessing the significance of global and local correlations
        under spatial autocorrelation: a nonparametric approach, Biometrics 2014.
    .. [2] Burt et al., Generative modeling of brain maps with spatial autocorrelation,
        NeuroImage, 2020.
    """
    x = array.flatten()
    d = distance_matrix(array)

    # Compute pairwise squared differences v_ij
    v_ij = 0.5 * (x[:, None] - x[None, :]) ** 2

    # Compute Gaussian kernel weights w_ij
    s = np.abs(lag - d)
    w_ij = np.exp(-((2.68 * s) ** 2) / (2 * bw**2))

    # Compute weighted sums
    mask = np.triu(np.ones_like(d), k=1)
    numerator = np.sum(w_ij * v_ij * mask)
    denominator = np.sum(w_ij * mask)

    return numerator / denominator if denominator != 0 else np.nan


def distance_matrix(array: np.ndarray) -> np.ndarray:
    """Pairwise euclidean distance for elements of a 2D array.

    Args:
        array: 2D array of values.

    Returns:
        Squared distance matrix.
    """
    rows, cols = array.shape
    coords = np.column_stack([np.indices((rows, cols)).reshape(2, -1).T])
    return squareform(pdist(coords, metric="euclidean"))


class Mask:
    """Generate a random 2D Binary mask for an input array.

    Attributes:
        array: 2D array as reference.
        perc: Percentage [0, 1] defining how many pixels to mask in the array.

    """

    def __init__(self, array: np.ndarray, perc: float = 0.2) -> None:
        self.array = array
        self.perc = perc
        self.size = self.array.size
        self.mask: np.ndarray | None = None

    @property
    def _init_mask(self) -> np.ndarray:
        """Initialize mask array."""
        return np.zeros_like(self.array, dtype=bool)

    @property
    def patch(self) -> np.ndarray:
        """Create a binary mask by defining a square within the array with random
        position (x0, y0). The size is defined by the set percentage of masked pixels
        within the array."""
        _length = np.round(np.sqrt(self.perc * self.size)).astype(int)
        x0 = np.random.randint(0, np.shape(self.array)[0] - _length)
        y0 = np.random.randint(0, np.shape(self.array)[1] - _length)
        self.mask = self._init_mask
        self.mask[x0 : x0 + _length, y0 : y0 + _length] = 1
        return self.mask

    @property
    def scatter(self) -> np.ndarray:
        """Create a binary mask by defining random pixels within the array. The number
        of masked pixels is defined by the set percentage of masked pixels within the
        array."""
        num_keep = int(self.perc * self.size)
        keep_indices = np.random.choice(self.size, num_keep, replace=False)
        self.mask = self._init_mask
        self.mask.ravel()[keep_indices] = True
        return self.mask

    @property
    def coordinates(self) -> np.ndarray | None:
        """Get x-, and y-coordinates of mask values."""
        if self.mask is None:
            return None
        return np.argwhere(self.mask)


def permute(array: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Permute a 2D array by in-place shuffling. Optionally, a mask array can be used
    that omits shuffling for all mask pixels.

    Args:
        array: 2D array of values.
        mask: 2D mask with elements to exclude during shuffling. Defaults to None.

    Returns:
        Permuted array.
    """
    if mask is None:
        mask = np.zeros_like(array, dtype=bool)

    # find pixels to permute (False in mask means permute)
    permute_indices = np.flatnonzero(mask == 0)
    permute_values = array.ravel()[permute_indices]

    # shuffle in-place
    np.random.shuffle(permute_values)
    array.ravel()[permute_indices] = permute_values

    return array
