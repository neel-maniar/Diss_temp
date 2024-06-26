import gpjax as gpx

from typing import (
    TypeVar,
)
import cola
import jax.numpy as jnp
from jaxtyping import (
    Num,
)
from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.kernels.base import AbstractKernel
from gpjax.likelihoods import (
    AbstractLikelihood,
    Gaussian,
    NonGaussian,
)
from gpjax.mean_functions import AbstractMeanFunction
from gpjax.typing import (
    Array,
)


class CustomPosterior(gpx.gps.ConjugatePosterior):
    """
    Custom Posterior class which inherits from GPJax conjugate posterior, but
    does not use cola.solve() method and uses jnp.solve() instead.
    """

    def predict(
        self,
        test_inputs: Num[Array, "N D"],
        train_data: Dataset,
    ) -> GaussianDistribution:
        """
        Same as the parent class method, but uses jnp.solve() instead of cola.solve().

        Parameters
        ----------
        test_inputs (Num[Array, "N D"]):
            A Jax array of test inputs at which the predictive distribution is evaluated.
        train_data (Dataset):
            A `gpx.Dataset` object that contains the input and output data used for training dataset.

        Returns
        -------
        GaussianDistribution:
            A function that accepts an input array and returns the predictive distribution as a `GaussianDistribution`.
        """
        # Unpack training data
        x, y = train_data.X, train_data.y

        # Unpack test inputs
        t = test_inputs

        # Observation noise o²
        obs_noise = self.likelihood.obs_stddev**2
        mx = self.prior.mean_function(x)

        # Precompute Gram matrix, Kxx, at training inputs, x
        Kxx = self.prior.kernel.gram(x)
        Kxx += cola.ops.I_like(Kxx) * self.jitter

        # Σ = Kxx + Io²
        Sigma = Kxx + cola.ops.I_like(Kxx) * obs_noise
        Sigma = cola.PSD(Sigma)

        mean_t = self.prior.mean_function(t)
        Ktt = self.prior.kernel.gram(t)
        Kxt = self.prior.kernel.cross_covariance(x, t)

        Sigma_jnp = Sigma.to_dense()
        Sigma_inv_Kxt = jnp.linalg.solve(Sigma_jnp, Kxt)
        # Sigma_inv_Kxt = cola.solve(Sigma, Kxt)

        # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
        mean = mean_t + jnp.matmul(Sigma_inv_Kxt.T, y - mx)

        # Ktt  -  Ktx (Kxx + Io²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        covariance += cola.ops.I_like(covariance) * self.prior.jitter
        covariance = cola.PSD(covariance)

        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)
