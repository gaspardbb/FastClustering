from abc import ABC, abstractmethod
from typing import Union, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from scipy.stats import rv_continuous

from _clustering import VoseAlias


class Distribution(ABC):
    """Mathematical distribution. Can be sampled from."""

    @abstractmethod
    def sample(self, n: Union[int, Iterable[int]], **kwargs) -> np.ndarray:
        """Sample from the distribution. Samples have shape: (n, ...)."""
        pass

    def has_same_dim(self, other: "Distribution"):
        """Take one sample from two distributions and check if they have same dimension."""
        sample_this = self.sample(1)
        sample_other = other.sample(1)
        return sample_this.shape[-1] == sample_other.shape[-1]

    def plot(self, n_samples, ax=None, **kwargs):
        """Plot samples from the distribution."""
        if ax is None:
            ax = plt.gca()
        ax: plt.Axes

        X = self.sample(n_samples)[:, :2]
        ax.scatter(*X.T, **kwargs)

        return ax


class UnstationnaryDistribution(Distribution, ABC):
    """Mathematical distribution, which evolves with time."""

    @abstractmethod
    def step(self, n: int = 1):
        pass


class OscillatingGaussian(UnstationnaryDistribution):

    def __init__(self, period: int, dim: int = 2, amplitude: float = 1., 
                 direction: np.ndarray = None, initial_pos: np.ndarray = None, 
                 covariance: float = 1.):
        if direction is None:
            direction = np.ones(dim)
        if initial_pos is None:
            initial_pos = np.zeros(dim)
        assert direction.ndim == initial_pos.ndim == 1
        assert direction.size == initial_pos.size == dim

        self._direction = direction 
        self._amplitude = amplitude
        self._initial_pos = initial_pos
        self._covariance = covariance
        
        self.period = period
        self.dim = dim
        self.current_step = 0

    @property
    def mean(self):
        """Mean of the distribution at step t."""
        fraction = self.current_step / self.period
        return self._initial_pos + np.math.sin(2 * np.pi * fraction) * self._amplitude * self._direction
    
    @property
    def covariance(self):
        """Covariance of the distribution at step t. Can be overriden by subclasses."""
        return np.eye(self.dim) * self._covariance
    
    def step(self, n: int = 1):
        self.current_step += n

    def sample(self, n: int = 1, **kwargs):
        return np.random.multivariate_normal(self.mean, self.covariance, size=n)
    


class FiniteDistribution(Distribution):
    """Distribution whose support is a finite, discrete point cloud."""

    def __init__(self, X: np.ndarray, weights: Optional[np.ndarray] = None, replace=False):
        """
        Constructs a FiniteDistribution.

        Parameters
        ----------
        replace : bool
            Whether to sample with or without replacement.
        X:
            Support of the distribution.
        weights:
            Histogram of the point cloud. If None, it is set to uniform.
        """
        assert X.ndim == 2, f"The support of your distribution should be a 2 dim array (samples, dim). Got {X.ndim}."
        self.n_samples, self.dim = X.shape
        self.support = X.copy()

        self.replace = replace
        if weights is None:
            # If weights are unspecified, take uniform distribution and use Numpy's choice method.
            self.weights = np.ones(self.n_samples) / self.n_samples
            self._sampler = lambda size: np.random.choice(a=self.n_samples, size=size, replace=self.replace, p=None)
        else:
            assert weights.shape == (self.n_samples, ), f"The weights of your distribution does not have the right shape. " \
                                                        f"Expected {(self.n_samples, )}, got {weights.shape}."
            self.weights = weights
            if self.replace:
                # Use Vose efficient sampling if with replacement
                self._vose_sampler = VoseAlias(self.weights)
                self._sampler = self._vose_sampler.sample
            else:
                self._sampler = lambda size: np.random.choice(a=self.n_samples, size=size, replace=self.replace,
                                                              p=weights)

    def __repr__(self):
        return f"<Discrete distribution with {self.n_samples} points>"

    def sample(self, n: Union[int, Iterable[int]], **kwargs) -> np.ndarray:
        if n > self.n_samples and (not self.replace):
            raise ValueError(f"Your distribution has {self.n_samples} points in its support, but you want to sample "
                             f"{n} points.")
        if n == -1:
            n = self.n_samples

        result = self.support[self._sampler(n)]
        if result.ndim == 1:
            return result.reshape(1, -1)
        return result

    def plot(self, n_samples=None, prefactor_s: float = 1., ax=None, **kwargs):
        """
        Plot all the support of the distribution.

        Parameters
        ----------
        n_samples:
            Number of samples to plot. If None, all the points of the distributions are shown.
        prefactor_s:
            Size of the points depend on the weight of the histogram. Use this to make them bigger/smaller.
        """
        if n_samples is None:
            n_samples = self.n_samples
        super(FiniteDistribution, self).plot(n_samples=n_samples, s=self.weights[:n_samples] * prefactor_s, ax=ax, **kwargs)


class ContinuousDistribution(Distribution):

    def __init__(self, sampler):
        """A continuous distribution implements a sample method."""
        self._sampler = sampler

    def sample(self, n: Union[int, Iterable[int]], **kwargs) -> np.ndarray:
        return self._sampler(n)

    @staticmethod
    def from_scipystats(f: rv_continuous, **params):
        """From a scipystats random variable object, return a ContinuousDistribution."""
        def sampler(n):
            return f.rvs(size=n, **params)
        return ContinuousDistribution(sampler)

    @staticmethod
    def random_gaussian_mixture(dim=2, n_gaussians=5, tau=1):
        """
        Returns a ContinuousDistribution which is a mixture of gaussian.

        Parameters
        ----------
        dim:
            Distribution defined on R^dim.
        n_gaussians:
            Number of components.
        tau:
            Isotropic covariance dilatation factor.

        Notes
        -----
        Means are sampled in 0, 1 uniformly.
        """
        mus = np.random.random_sample((n_gaussians, dim))
        sigmas = np.stack([np.eye(dim) * tau] * n_gaussians)
        p = np.random.random_sample(n_gaussians)
        p /= p.sum()
        return ContinuousDistribution.gaussian_mixture(p, mus, sigmas)

    @staticmethod
    def gaussian_mixture(p, mus, sigmas):
        """Returns a ContinuousDistribution which is a mixture of Gaussian."""
        assert mus.ndim == (sigmas.ndim - 1) == 2
        def sampler(n):
            n_samples_comp = np.random.multinomial(n, p)
            X = np.vstack([np.random.multivariate_normal(mean, covariance, sample) for (mean, covariance, sample) in
                           zip(mus, sigmas, n_samples_comp)])
            # y = np.concatenate([np.full(sample, j, dtype=int) for j, sample in enumerate(n_samples_comp)])
            return X
        return ContinuousDistribution(sampler)

    @staticmethod
    def uniform(low: float, high: float, dim: int = 3):
        def sampler(n):
            return np.random.uniform(low, high, (n, dim))
        return ContinuousDistribution(sampler)

    @staticmethod
    def gaussian(mu, sigma):
        """Returns a ContinuousDistribution which is a simple Multivariate normal."""
        assert mu.ndim == (sigma.ndim - 1) == 1
        def sampler(n):
            return np.random.multivariate_normal(mu, sigma, size=n)
        return ContinuousDistribution(sampler)

    def to_discrete(self, n_samples, **kwargs):
        return FiniteDistribution(self.sample(n_samples), **kwargs)


def squared_bures(sigma_1, sigma_2):
    """
    Squared Bures metric between two covariance matrices.

    Parameters
    ----------
    sigma_1, sigma_2: np.ndarray
        Two pd matrices.

    Returns
    -------
    distance: float
        Bures' distance.
    """
    sqrt_sigma_1 = sqrtm(sigma_1)
    matrix = sigma_1 + sigma_2 - 2 * sqrtm(sqrt_sigma_1 @ sigma_2 @ sqrt_sigma_1)
    return np.sum(np.diag(matrix))


def ot_gaussian(mu_alpha, mu_beta, sigma_alpha, sigma_beta):
    """Return the **squared** 2-wasserstein distance."""
    return np.linalg.norm(mu_alpha - mu_beta, 2) ** 2 + squared_bures(sigma_alpha, sigma_beta)