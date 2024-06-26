"""
This module contains functions for Gaussian process (GP) inference, including initializing a GP model, optimizing the marginal log-likelihood (MLL), computing the posterior distribution, and making predictions.

Functions:
- initialise_gp: Initialize a Gaussian Process (GP) model for inference.
- optimise_mll: Optimize the marginal log-likelihood (MLL) of a Gaussian process model.
- get_posterior: Compute the posterior distribution of a Gaussian process given a kernel and training dataset.
- predict: Predict the mean and standard deviation of the target variable using the optimized posterior distribution.
- steal_diag_params: Steal diagonal parameters from a given dataset collection and apply them to a Gaussian process.
"""

from jax import config

config.update("jax_enable_x64", True)
import sys

from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jaxtyping import install_import_hook
import optax as ox
from matplotlib import rcParams

sys.path.append("src")

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

from gpjax.gps import ConjugatePosterior
from gpjax.mean_functions import AbstractMeanFunction
from gpjax import Dataset

import tensorflow_probability.substrates.jax.bijectors as tfb
from gpjax.base import param_field

# Enable Float64 for more stable matrix inversions.

from utils.CustomPosterior import CustomPosterior


# def initialise_gp(
#     kernel,
#     mean: AbstractMeanFunction,
#     dataset: Dataset,
#     initial_sigma_n: jnp.float64 = 2 * 1e-2,
# ) -> ConjugatePosterior:
#     """
#     Initialize a Gaussian Process (GP) model for inference.

#     Parameters
#     ----------
#     kernel : Any
#         The kernel function for the GP model.
#     mean : AbstractMeanFunction
#         The mean function for the GP model.
#     dataset : Dataset
#         The dataset used for training the GP model.
#     initial_sigma_n : float, optional
#         The initial value for the observation noise standard deviation.

#     Returns
#     -------
#     ConjugatePosterior
#         The posterior GP model.
#     """
#     prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)
#     likelihood = gpx.likelihoods.Gaussian(
#         num_datapoints=dataset.n,
#         obs_stddev=jnp.array(
#             [initial_sigma_n], dtype=jnp.float64
#         ),  # TODO check what sigma_n should be
#     )
#     posterior = prior * likelihood
#     return posterior


def initialise_gp(
    kernel_class,
    mean_func: AbstractMeanFunction,
    dataset: Dataset,
    input_dim,
    initial_params=jnp.array([1e-2, 1.0, 1.0], dtype=jnp.float64),
) -> ConjugatePosterior:
    """
    Initialize a Gaussian Process (GP) model for inference.

    Parameters
    ----------
    kernel : Any
        The kernel function for the GP model.
    mean : AbstractMeanFunction
        The mean function for the GP model.
    dataset : Dataset
        The dataset used for training the GP model.
    initial_sigma_n : float, optional
        The initial value for the observation noise standard deviation.

    Returns
    -------
    ConjugatePosterior
        The posterior GP model.
    """
    active_dims = jnp.arange(input_dim, dtype=jnp.int32)
    
    if initial_params is not None:
        initial_sigma_n, initial_lengthscale, initial_variance = initial_params

        kernel = kernel_class(
            kernel=gpx.kernels.RBF(
                active_dims=active_dims,
                lengthscale=initial_lengthscale,
                variance=initial_variance,
            )
        )
    else:
        initial_sigma_n = 2 * 1e-2
        kernel = kernel_class()

    prior = gpx.gps.Prior(mean_function=mean_func, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=dataset.n,
        obs_stddev=jnp.array(
            [initial_sigma_n], dtype=jnp.float64
        ),  # TODO check what sigma_n should be
    )
    posterior = prior * likelihood
    return posterior


def optimise_mll(
    posterior: ConjugatePosterior,
    dataset,
    NIters=500,
    verbose=True,
    optimisation="scipy",
    key=jr.key(0),
):
    """
    Optimize the marginal log-likelihood (MLL) of a Gaussian process model.

    Parameters:
        posterior (ConjugatePosterior): The posterior distribution of the Gaussian process model.
        dataset: The training dataset.
        NIters (int): The maximum number of iterations for optimization. Default is 500.
        verbose (bool): Whether to print verbose output during optimization. Default is True.
        optimisation (str): The optimization method to use. Can be "scipy" or "adam". Default is "scipy".
        key: The random key for the optimization. Default is jr.key(0).

    Returns:
        opt_posterior (ConjugatePosterior): The optimized posterior distribution.
        last_history (dict): The history of the optimization process.

    """

    # define the MLL using dataset_train
    objective = gpx.objectives.ConjugateMLL(negative=True)

    # Optimise to minimise the MLL
    if optimisation == "scipy":
        opt_posterior, history = gpx.fit_scipy(
            model=posterior,
            objective=objective,
            train_data=dataset,
            max_iters=NIters,
            verbose=verbose,
        )

    elif optimisation == "adam":
        opt_posterior, history = gpx.fit(
            model=posterior,
            objective=objective,
            train_data=dataset,
            num_iters=NIters,
            verbose=verbose,
            optim=ox.adam(2e-1),
            key=key,
        )

    if verbose:
        _params = ravel_pytree(posterior)[0].tolist()
        _params[2] = _params[2] ** 0.5
        objective = gpx.objectives.ConjugateMLL(negative=True)
        print("Initial: σ_n, l, σ_f:", _params)
        print("Initial NMLL:", objective(posterior, dataset))
        _params = ravel_pytree(opt_posterior)[0].tolist()
        _params[2] = _params[2] ** 0.5
        print("Final: σ_n, l, σ_f:", _params)
        print("Final NMLL:", objective(opt_posterior, dataset))

    return opt_posterior, history[-1]


def get_posterior(
    kernel,
    dataset_train,
    verbose=False,
    optimisation="scipy",
    key=jr.key(0),
    noise=2 * 1e-2,
    NIters=500,
):
    """
    Compute the posterior distribution of a Gaussian process given a kernel and training dataset.

    Parameters
    ----------
        kernel (gpx.kernels.Kernel): The kernel function to use for the Gaussian process.
        dataset_train (tuple): The training dataset, consisting of input features and corresponding target values.
        verbose (bool, optional): Whether to print additional information during the computation. Defaults to False.
        optimisation (str, optional): The optimization method to use. Defaults to "scipy".
        key (jnp.ndarray, optional): The random key for reproducibility. Defaults to jr.key(0).
        noise (float, optional): The noise level of the Gaussian process. Defaults to 2 * 1e-2.
        NIters (int, optional): The number of optimization iterations. Defaults to 500.

    Returns
    -------
        tuple: A tuple containing the optimized posterior distribution, the optimized marginal log-likelihood,
               and the optimized parameters of the posterior distribution.
    """
    meanf = gpx.mean_functions.Zero()
    posterior = initialise_gp(kernel, meanf, dataset_train, noise)
    opt_posterior, opt_MLL = optimise_mll(
        posterior,
        dataset_train,
        verbose=verbose,
        optimisation=optimisation,
        key=key,
        NIters=NIters,
    )

    if verbose:
        _params = ravel_pytree(posterior)[0].tolist()
        _params[2] = _params[2] ** 0.5
        objective = gpx.objectives.ConjugateMLL(negative=True)
        print("Initial: σ_n, l, σ_f:", _params)
        print("Initial MLL:", objective(posterior, dataset_train))
        _params = ravel_pytree(opt_posterior)[0].tolist()
        _params[2] = _params[2] ** 0.5
        print("Final: σ_n, l, σ_f:", _params)
        print("Final MLL:", objective(opt_posterior, dataset_train))

    return opt_posterior, opt_MLL, ravel_pytree(opt_posterior)[0]


def predict(opt_posterior, test_X, dataset_train):
    """
    Predicts the mean and standard deviation of the target variable using the optimized posterior distribution.

    Parameters
    ----------
        opt_posterior (object):
            The optimized posterior distribution.
        test_X (array-like):
            The input data for prediction.
        dataset_train (array-like):
            The training dataset used to optimize the posterior distribution.

    Returns:
        tuple:
            A tuple containing the predicted mean and standard deviation.

    """
    opt_posterior = CustomPosterior(opt_posterior.prior, opt_posterior.likelihood)
    dist = opt_posterior.predict(test_X, train_data=dataset_train)
    pred_mean = dist.mean()
    pred_std = dist.stddev()
    return pred_mean, pred_std


def steal_diag_params(diag_params, dataset_coll_train, kernel_class):
    """
    Steals diagonal parameters from a given dataset collection and applies them to a Gaussian process.

    Parameters
    ----------
    - diag_params (numpy.ndarray):
        The diagonal parameters to be applied to the Gaussian process.
    - dataset_coll_train (DatasetCollection):
        The training dataset collection.
    - kernel (GPy.kern.Kern):
        The kernel function for the Gaussian process.

    Returns
    -------
    - opt_art_posterior (GPy.models.GPRegression):
        The optimized posterior Gaussian process with the stolen diagonal parameters.
    """
    meanf = gpx.mean_functions.Zero()
    input_dim = dataset_coll_train.X[-1,-1]
    art_posterior = initialise_gp(kernel_class, meanf, dataset_coll_train, input_dim, None)
    ravel_func = ravel_pytree(art_posterior)[1]
    opt_art_posterior = ravel_func(diag_params)
    return opt_art_posterior
