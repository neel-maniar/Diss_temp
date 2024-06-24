from jax import config

config.update("jax_enable_x64", True)
import sys
from dataclasses import dataclass

from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from jax import grad, hessian, jit, vmap
from jaxtyping import Array, Float, install_import_hook
import optax as ox
from matplotlib import rcParams
import scipy.io as sio

sys.path.append("src")
from kernels.DiagonalKernel3D import DiagonalKernel3D
from kernels.CurlFreeKernel import CurlFreeKernel
from kernels.ArtificialKernel3D import ArtificialKernel3D
from kernels.ArtificialKernelExplicit3D import ArtificialKernelExplicit3D
from utils.data_tools import (
    generate_data,
    transform_data,
    regular_train_points,
    add_collocation_points,
)
from utils.performance import rmse
from utils.plotting_tools import plot_data, plot_pred
from line_profiler import LineProfiler

profiler = LineProfiler()

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

from time import time


# Enable Float64 for more stable matrix inversions.
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
colors = rcParams["axes.prop_cycle"].by_key()["color"]


def initialise_gp(kernel, mean, dataset, initial_sigma_n=2 * 1e-2):
    prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=dataset.n,
        obs_stddev=jnp.array(
            [initial_sigma_n], dtype=jnp.float64
        ),  # TODO check what sigma_n should be
    )
    posterior = prior * likelihood
    return posterior


def optimise_mll(
    posterior, dataset, NIters=500, verbose=True, optimisation="scipy", key=jr.key(0)
):
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
            optim=ox.adam(1e-1),
            key=key,
        )

    if verbose:
        print("Length of history: ", len(history))
    return opt_posterior


def get_posterior(
    kernel,
    dataset_train,
    verbose=False,
    params=False,
    optimisation="scipy",
    key=jr.key(0),
):
    """Predicts the latent function at the test points given the training data and the kernel.

    Args:
        kernel (gpx kernel): Kernel
        dataset_train (jnp.ndarray): Training data
        test_X (jnp.ndarray): Test inputs

    Returns:
        jnp.ndarray: Output of the latent function at the test points
    """
    meanf = gpx.mean_functions.Zero()
    posterior = initialise_gp(kernel, meanf, dataset_train)
    opt_posterior = optimise_mll(
        posterior, dataset_train, verbose=verbose, optimisation=optimisation, key=key
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

    if params:
        return opt_posterior, ravel_pytree(opt_posterior)[0]
    return opt_posterior


def latent_distribution(opt_posterior, pos_3d, dataset_train):
    latent = opt_posterior.predict(pos_3d, train_data=dataset_train)
    latent_mean = latent.mean()
    return latent_mean


def predict(opt_posterior, test_X, dataset_train, output_dim=2):
    pred_mean = latent_distribution(opt_posterior, test_X, dataset_train)
    function_pred = pred_mean[test_X[:, output_dim] != output_dim].reshape(
        -1, output_dim, order="F"
    )
    return function_pred


def steal_diag_params(diag_params, dataset_coll_train, kernel):
    meanf = gpx.mean_functions.Zero()
    art_posterior = initialise_gp(kernel, meanf, dataset_coll_train)
    ravel_func = ravel_pytree(art_posterior)[1]
    opt_art_posterior = ravel_func(diag_params)
    return opt_art_posterior
