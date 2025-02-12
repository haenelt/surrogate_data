# -*- coding: utf-8 -*-
"""Creation of random gaussian fields for data simulation."""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import gamma

__all__ = ["RandomGaussianField", "MaternRandomGaussianField"]


class RandomGaussianField:
    """Generate a 2D Random Gaussian Field.

    The class generates isotropic and homogeneous 2D random fields. The spatial
    autocorrelation thereby only depends on the distance between points and can be
    expressed by a spectral density function in Fourier space. Furthermore, since the
    random gaussian field is isotropic, the spectral density only depends on the
    k-space vector magnitude. In the following implementation, two spectral density
    functions are provided: RandomGaussianField._matern and RandomGaussianField._power.

    To simulate random Gaussian fields, a continuous white noise Gaussian process with
    zero mean and unit variance is colored by multiplying its representation in Fourier
    space by a spectral density function.

    For the specific implementation, which closely follows [1, 2], a 2D array with data
    sampled randomly from a standard normal distribution is created and its
    representation in Fourier space is computed. For each array element, the shifted
    Fourier component k is used to calculate the amplitude of a spectral density
    function. Note that the mean shift was removed by setting the spectral amplitude for
    k = 0 to zero. Both arrays are multiplied in Fourier space and the inverse Fourier
    transform is taken. The final array is normalized such that it has zero mean and
    unit variance.

    Attributes:
        n: Array size in both dimensions.
        seed: Set seed of random number generator for reproducible results. Defaults
            to None.

    References:
    .. [1] Burt et al., Generative modeling of brain maps with spatial autocorrelation,
        NeuroImage, 2020.
    .. [2] https://github.com/bsciolla/gaussian-random-fields/blob/master/
        gaussian_random_fields.py

    """

    alpha: float = 3.0  # Exponent for power law kernel.

    def __init__(self, n: int, seed: int | None = None) -> None:
        self.n = n
        self.gfield: np.ndarray | None = None  # initialize random field
        np.random.seed(seed)

    @property
    def spat_freq(self) -> np.ndarray:
        """Computation of spatial frequency array.

        Returns:
            Magnitude of shifted k-space vector.
        """
        k = np.mgrid[: self.n, : self.n] - int((self.n + 1) / 2)
        k = np.fft.fftshift(k)
        k = np.sqrt(k[0] ** 2 + k[1] ** 2) + 1e-10

        return k

    @property
    def kernel(self) -> np.ndarray:
        """Power spectral density function. Note that the mean shift was removed at
        k = 0.

        Returns:
            Array with correlation described by a law for given spatial frequencies.
        """
        amplitude = np.power(self.spat_freq, -self.__class__.alpha / 2.0)
        amplitude[0, 0] = 0
        return amplitude

    def noise(self, mu: float, sigma: float) -> np.ndarray:
        """Definition of the continuous white noise Gaussian process.

        Args:
            mu: Expectation value.
            sigma: Standard devation.

        Returns:
            Complex gaussian random noise with normal distribution.
        """
        noise_real = np.random.normal(mu, sigma, size=(self.n, self.n))
        noise_imag = np.random.normal(mu, sigma, size=(self.n, self.n))
        return noise_real + 1j * noise_imag

    def sample(self) -> np.ndarray:
        """Color white noise Gaussian process to sample random Gaussian fields with
        defined spatial covariance structure. The white noise process with zero mean and
        unit variance is colored by multiplying with the spectral density function. The
        resulting array is Fourier transformed and normalized.

        Returns:
            Normalized random Gaussian field.
        """
        gfield = np.fft.ifft2(self.noise(0, 1) * self.kernel).real
        gfield = gfield - np.mean(gfield)
        gfield = gfield / np.std(gfield)
        self.gfield = gfield

        return gfield

    def save_image(self, fname: str) -> None:
        """The resulting random Gaussian field is scaled and saved as image to disk."""
        # check if random gaussian field was already created
        if self.gfield is None:
            raise ValueError("No Random Gaussian Field was sampled!")
        # scale
        arr = self.gfield + np.abs(np.min(self.gfield))
        arr = arr / np.max(arr) * 255.0
        image = Image.fromarray(arr.astype(np.uint8))
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        image.save(fname)


class MaternRandomGaussianField(RandomGaussianField):
    """Generate a 2D Random Gaussian Field with Matern kernel.

    The class generates isotropic and homogeneous 2D random fields. The only difference
    to the RandomGaussianField implementation is the usage of the Matern covariance
    function as spectral density function.

    Attributes:
        n: Array size in both dimensions.
        seed: Set seed of random number generator for reproducible results. Defaults
            to None.

    """

    phi: float = 0.3  # Range parameter for matern kernel.
    kappa: float = 0.5  # Smoothness parameter for matern kernel.

    @property
    def kernel(self) -> np.ndarray:
        """Computes the Matern correlation function for given distances and parameters
        in its representation in Fourier space. The formula is taken from [1]. Note that
        the mean shift was removed at k = 0.

        Returns:
            Array with correlation described by the Matern covariance function for given
            spatial frequencies.

        References:
        .. [1] https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
        """
        phi = self.__class__.phi
        kappa = self.__class__.kappa
        if phi <= 0:
            raise ValueError("Phi must be greater than 0!")
        if kappa <= 0:
            raise ValueError("Kappa must be greater than 0!")

        n = 2
        a = 2**n * np.pi ** (n / 2) * gamma(kappa + n / 2) * (2 * kappa) ** kappa
        b = gamma(kappa) * phi ** (2 * kappa)
        c = (2 * kappa / phi**2 + 4 * np.pi**2 * self.spat_freq**2) ** (
            -(kappa + n / 2)
        )
        amplitude = np.array(a / b * c)
        amplitude[0, 0] = 0
        return amplitude
