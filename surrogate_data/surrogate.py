# -*- coding: utf-8 -*-
"""Compute surrogate maps."""

import warnings
from functools import cached_property

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

__all__ = [
    "SurrogateMap",
    "variogram",
    "distance_matrix",
    "permute",
]


def _exponential_kernel(distances: np.ndarray) -> np.ndarray:
    """Normalized exponential decay kernel."""
    return np.exp(-distances / np.max(distances, axis=-1, keepdims=True))


def _gaussian_kernel(distances: np.ndarray) -> np.ndarray:
    """Normalized gaussian decay kernel."""
    return np.exp(
        -1.25 * np.square(distances / np.max(distances, axis=-1, keepdims=True))
    )


def _uniform_kernel(distances: np.ndarray) -> np.ndarray:
    """Normalized uniform kernel."""
    return np.ones(distances.shape) / distances.shape[-1]


_kernel: dict = {
    "exp": _exponential_kernel,
    "gaussian": _gaussian_kernel,
    "uniform": _uniform_kernel,
}


class SurrogateMap:
    """Generate 2D surrogate maps.

    The class computes surrogate maps of a 2D input array with preserved spatial
    autocorrelation. It closely follows the implementation in [1, 2].

    Attributes:
        array: 2D array of values.

    References:
    .. [1] Viladomat et al., Assessing the significance of global and local correlations
        under spatial autocorrelation: a nonparametric approach, Biometrics 2014.
    .. [2] Burt et al., Generative modeling of brain maps with spatial autocorrelation,
        NeuroImage, 2020.

    """

    # number of cores for parallel variogram computation
    DEFAULT_NUM_CORES: int = 4

    # bandwidth and list of lags for variogram computation
    DEFAULT_BW: float = 1.0
    DEFAULT_LAGS: np.ndarray = np.linspace(0.0, 8.0, 25)

    # distance-depending smoothing kernel (exp, gaussian, uniform)
    DEFAULT_KERNEL = "gaussian"

    # array of kernel sizes for data smoothing
    DEFAULT_K_SMOOTH: np.ndarray = np.arange(2, 100, 5)

    def __init__(self, array: np.ndarray, **kwargs) -> None:
        self.array = array

        # Defaults parameters for smoothing and variogram fitting
        self.lags: np.ndarray = kwargs.get("lags", self.DEFAULT_LAGS)
        self.bw: float = kwargs.get("bw", self.DEFAULT_BW)
        self.k_smooth: np.ndarray = kwargs.get("k_smooth", self.DEFAULT_K_SMOOTH)
        self.kernel: str = kwargs.get("kernel", self.DEFAULT_KERNEL)
        self.num_cores: int = kwargs.get("num_cores", self.DEFAULT_NUM_CORES)

        # Initialize fitting parameters
        # alpha: Variogram fit parameter (intercept)
        # beta: Variogram fit paramter (slope)
        # k: Kernel neighborhood size for best variogram fit
        self.alpha_best: float | None = None
        self.beta_best: float | None = None
        self.k_best: int | None = None

    @property
    def permute(self) -> np.ndarray:
        """Permute input array."""
        return permute(self.array)

    @property
    def noise(self) -> np.ndarray:
        """Make Gaussian noise with unit variance needed for generation of final
        surrogate maps."""
        height, width = np.shape(self.array)
        _noise = np.random.normal(0.0, 1.0, size=(height, width))
        return _noise

    @cached_property
    def distances(self) -> np.ndarray:
        """Square distance matrix of input array."""
        return distance_matrix(self.array)

    @cached_property
    def var_0(self) -> np.ndarray:
        """Variogram of input array."""
        return variogram(self.array, self.lags, self.bw, self.num_cores)

    def fit(self) -> dict:
        """Perform linear regression on permuted data smoothed with different kernel
        sizes to find the smoothing kernel that best matches the variogram curve of the
        input array.

        Returns:
            Dictionary containing the fit parameters from all smoothing kernels and from
            the best fit.
        """
        # (1) permute data
        arr_x = self.permute

        alphas = []
        betas = []
        residuals = []
        for k in tqdm(self.k_smooth):
            # (2) smooth and scale permuted data
            arr_smooth = self._smooth(arr_x, k)
            arr_smooth = self._scale(arr_smooth)
            # compute variogram
            var_x = variogram(arr_smooth, self.lags, self.bw, self.num_cores)

            alpha, beta, sse = self._fit(var_x, self.var_0)
            alphas.append(alpha)
            betas.append(beta)
            residuals.append(sse)

        # (3) get best fit parameters
        best_param = np.argmin(residuals)
        if best_param == len(residuals) - 1:
            warnings.warn(
                "The end of the neighborhood array was reached. I would recommend to "
                "change the kernel and/or increase the neighborhood size."
            )

        self.alpha_best = alphas[best_param]
        self.beta_best = betas[best_param]
        self.k_best = self.k_smooth[best_param]

        print(f"ALPHA: {self.alpha_best}")
        print(f"BETA: {self.beta_best}")
        print(f"K: {self.k_best}")

        return {
            "all": [alphas, betas, residuals],
            "best": [self.alpha_best, self.beta_best, self.k_best],
        }

    def generate(self, n_surr: int) -> np.ndarray:
        """Generate a number (n_surr) of surrogate maps. Maps can only be generated
        after variogram fitting."""
        res = Parallel(n_jobs=self.num_cores)(
            delayed(self._generate)() for _ in range(n_surr)
        )
        return np.array(res)

    def _smooth(self, arr: np.ndarray, k: int) -> np.ndarray:
        """Smoothes a 2D array by local kernel-weighted sum of array values. Different
        kernels can be selected.

        Args:
            arr: 2D array of values.
            k: Number of nearest neighbors to consider for smoothing.

        Returns:
            Smoothed array.
        """
        if k < 2:
            raise ValueError("Minimum k is 2!")

        height, width = np.shape(arr)
        arr_flat = arr.ravel()

        # find nearest neighbors
        sorted_idx = np.argsort(self.distances, axis=1)[:, :k]  # Indices of knn
        sorted_dist = np.take_along_axis(self.distances, sorted_idx, axis=1)  # (N, k)
        neighbor_vals = np.take_along_axis(
            arr_flat[:, None], sorted_idx, axis=0
        )  # (N, k)

        # kernel weights
        K = _kernel[self.kernel](sorted_dist)  # (N, k)

        # compute weighted sum (vectorized)
        smoothed_flat = np.sum(K * neighbor_vals, axis=1) / np.sum(K, axis=1)
        arr_smoothed = smoothed_flat.reshape(height, width)

        return arr_smoothed

    def _scale(self, arr: np.ndarray) -> np.ndarray:
        """Scale array to match min-max of input array.

        Args:
            arr: 2D array of values.

        Returns:
            Scaled array.
        """
        _min = np.min(arr)
        _max = np.max(arr)
        _min_0 = np.min(self.array)
        _max_0 = np.max(self.array)
        arr_scaled = (arr - _min) / (_max - _min) * (_max_0 - _min_0) + _min_0

        return arr_scaled

    def _generate(self) -> np.ndarray:
        """Helper function to generate surrogate map after variogram fitting, called by
        the SurrogateMap.generate method for parallelization purposes."""
        if self.k_best is None or self.alpha_best is None or self.beta_best is None:
            raise ValueError(
                "Variogram fitting not performed! Please run first the fit method."
            )

        # (1) permute data
        arr_x = self.permute
        # (2) smooth and scale permuted data
        arr_smooth = self._smooth(arr_x, k=self.k_best)
        arr_smooth = self._scale(arr_smooth)
        return np.sqrt(np.abs(self.beta_best)) * arr_smooth + (
            np.sqrt(np.abs(self.alpha_best)) * self.noise
        )

    @staticmethod
    def _fit(x, y) -> tuple:
        """Helper function to perform linear regression.

        Args:
            x: x-axis coordinates should be the variogram of the shuffled and smoothed
                array.
            y: y-axis coordiantes should be the variogram of the input array.

        Returns:
            Tuple containing intercept (alpha), slope (beta) and sum of squares error
                (sse).
        """
        coeffs, sse, _, _, _ = np.polyfit(x, y, deg=1, full=True)
        beta, alpha = coeffs
        return alpha, beta, sse


def _gamma(array: np.ndarray, distance: np.ndarray, lag: float, bw: float) -> float:
    """Computes the variogram gamma(h) for a specific lag h using the element positions
    of a 2D array as spatial coordinates. This function computes gamma for the smoothed
    variogram as outlined in [1, 2].

    Args:
        array: 2D array of values.
        distance: Squared distance matrix.
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

    # Compute pairwise squared differences v_ij
    v_ij = 0.5 * (x[:, None] - x[None, :]) ** 2

    # Compute Gaussian kernel weights w_ij
    s = np.abs(lag - distance)
    w_ij = np.exp(-((2.68 * s) ** 2) / (2 * bw**2))

    # Compute weighted sums
    mask = np.triu(np.ones_like(distance), k=1)
    numerator = np.sum(w_ij * v_ij * mask)
    denominator = np.sum(w_ij * mask)

    return numerator / denominator if denominator != 0 else np.nan


def variogram(
    array: np.ndarray, lags: np.ndarray, bw: float, num_cores: int = 4
) -> np.ndarray:
    """Compute the smoothed variogram for a range of lags (h).

    Args:
        array: 2D array of values.
        lags: List of lags.
        bw: Bandwidth parameter controlling smoothness.
        num_cores: Number of cores for parallelization. Defaults to 4.

    Returns:
        Variance for different spatial lags.
    """
    _d = distance_matrix(array)
    res = Parallel(n_jobs=num_cores)(delayed(_gamma)(array, _d, _h, bw) for _h in lags)
    return np.array(res)


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


def permute(array: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Permute a 2D array by in-place shuffling. Optionally, a mask array can be used
    that omits shuffling for all mask pixels.

    Args:
        array: 2D array of values.
        mask: 2D mask with elements to exclude during shuffling. Defaults to None.

    Returns:
        Permuted array.
    """
    _array = array.copy()
    if mask is None:
        mask = np.zeros_like(array, dtype=bool)

    # find pixels to permute (False in mask means permute)
    permute_indices = np.flatnonzero(mask == 0)
    permute_values = _array.ravel()[permute_indices]

    # shuffle in-place
    np.random.shuffle(permute_values)
    _array.ravel()[permute_indices] = permute_values

    return _array
