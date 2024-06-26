from jax import config

config.update("jax_enable_x64", True)
import sys
import numpy as np
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import install_import_hook
from matplotlib import rcParams
from tqdm.auto import tqdm

sys.path.append("src")
from kernels.ArtificialKernel import ArtificialKernel
from kernels.DiagonalKernel import DiagonalKernel
from kernels.DivFreeKernel import DivFreeKernel
from utils.data_tools import (
    generate_data,
    transform_data,
    regular_train_points,
    add_collocation_points,
)
from utils.performance import rmse, header, nlpd
from utils.plotting_tools import plot_data, plot_pred
from utils.inference import (
    get_posterior,
    predict,
    steal_diag_params,
    initialise_gp,
    optimise_mll,
)
from jax.flatten_util import ravel_pytree

from os.path import join, exists
import os

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

import toml


def script_2D(params):
    A = params["a"]
    NOISE = params["noise"]
    N_TRAIN = params["N_train"]
    N_TEST_1D = params["N_test_1D"]
    N_C_LIST = params["N_c_list"]
    N_REPEAT = params["nrRepeat"]
    REGULAR = params["regular"]
    OPTIMISER = params["optimiser"].lower()
    TRAIN_ARTIFICIAL = params["train_artificial"]
    PLOT = params["plot"]
    NAME = params["name"]

    # Store the results in these arrays
    rmseDiag = np.zeros((N_REPEAT, 1))
    rmseCust = np.zeros((N_REPEAT, 1))
    rmseArtif = np.zeros((N_REPEAT, len(N_C_LIST)))
    optMLLDiag = np.zeros((N_REPEAT, 1))
    optMLLCust = np.zeros((N_REPEAT, 1))
    optMLLArtif = np.zeros((N_REPEAT, len(N_C_LIST)))
    nlpdDiag = np.zeros((N_REPEAT, 1))
    nlpdCust = np.zeros((N_REPEAT, 1))
    nlpdArtif = np.zeros((N_REPEAT, len(N_C_LIST)))

    initial_lengthscale = 0.5
    initial_variance = 0.5

    initial_params = jnp.array(
        [NOISE, initial_lengthscale, initial_variance], dtype=jnp.float64
    )

    if N_REPEAT == 1:
        verbose = True
    else:
        verbose = False

    objective = gpx.objectives.ConjugateMLL(negative=True)

    for i in tqdm(range(N_REPEAT), desc=f"Repeating experiment {N_REPEAT} times"):
        ## Set seed
        master_key = jr.key(i)

        ## Get data
        x, y, xtest, ytest = generate_data(master_key, A, N_TRAIN, N_TEST_1D, NOISE)

        if PLOT:
            plot_data(x, y, xtest, ytest, directory=NAME)
        if REGULAR:
            x, y = regular_train_points(master_key, a=A, n=N_TRAIN, noise=NOISE)
        dataset_train, dataset_test = transform_data(x, y, xtest, ytest)

        ## Diagonal Kernel
        kernel_name = "Diagonal Kernel"
        if verbose:
            header(kernel_name)

        meanf = gpx.mean_functions.Zero()
        posterior = initialise_gp(
            DiagonalKernel, meanf, dataset_train, 2, initial_params
        )
        opt_posterior, opt_MLL = optimise_mll(
            posterior,
            dataset_train,
            verbose=verbose,
            optimisation=OPTIMISER,
            key=master_key,
            NIters=1000,
        )

        ypred, y_std = predict(opt_posterior, dataset_test.X, dataset_train)

        if PLOT:
            plot_pred(
                x,
                y,
                xtest,
                ytest,
                ypred,
                kernel_name=kernel_name,
                directory=NAME,
                regular=REGULAR,
            )
        rmseDiag[i] = rmse(ytest, ypred).item()
        nlpdDiag[i] = nlpd(ypred, y_std, ytest)
        optMLLDiag[i] = opt_MLL.item()
        diag_params = ravel_pytree(opt_posterior)[0]
        if verbose:
            print(f"{kernel_name} rmse: {rmse(ytest, ypred).item()}")

        ## Divergence-free Kernel
        kernel_name = "Divergence-free Kernel"
        if verbose:
            header(kernel_name)
        posterior = initialise_gp(
            DivFreeKernel, meanf, dataset_train, 2, initial_params
        )
        opt_posterior, opt_MLL = optimise_mll(
            posterior,
            dataset_train,
            verbose=verbose,
            optimisation=OPTIMISER,
            key=master_key,
            NIters=1000,
        )
        optMLLCust[i] = opt_MLL.item()
        ypred, y_std = predict(opt_posterior, dataset_test.X, dataset_train)
        rmseCust[i] = rmse(ytest, ypred).item()
        nlpdCust[i] = nlpd(ypred, y_std, ytest)
        if PLOT:
            plot_pred(
                x,
                y,
                xtest,
                ytest,
                ypred,
                kernel_name=kernel_name,
                directory=NAME,
                regular=REGULAR,
            )
        if verbose:
            print(f"{kernel_name} rmse: {rmse(ytest, ypred).item()}")

        ## Artificial Kernel
        for j, N_c in enumerate(N_C_LIST):
            kernel_name = f"Artificial Kernel, $N_c$ = {N_c}"
            if verbose:
                header(kernel_name)
            dataset_train_artif = add_collocation_points(
                dataset_train, xtest, N_c, master_key, 1, REGULAR
            )
            if TRAIN_ARTIFICIAL:
                posterior = initialise_gp(
                    ArtificialKernel, meanf, dataset_train, 2, initial_params
                )
                opt_posterior, opt_MLL = optimise_mll(
                    posterior,
                    dataset_train,
                    verbose=verbose,
                    optimisation=OPTIMISER,
                    key=master_key,
                    NIters=1000,
                )
            else:
                opt_posterior = steal_diag_params(
                    diag_params, dataset_train_artif, ArtificialKernel
                )
                opt_MLL = objective(opt_posterior, dataset_train_artif)

            optMLLArtif[i, j] = opt_MLL.item()

            ypred, y_std = predict(opt_posterior, dataset_test.X, dataset_train_artif)
            rmseArtif[i, j] = rmse(ytest, ypred).item()
            nlpdArtif[i, j] = nlpd(ypred, y_std, ytest)

            if PLOT:
                plot_pred(
                    x,
                    y,
                    xtest,
                    ytest,
                    ypred,
                    kernel_name=kernel_name,
                    directory=NAME,
                    regular=REGULAR,
                )
            if verbose:
                print(f"{kernel_name} rmse:", rmse(ytest, ypred).item())

    # Define a dictionary to store the arrays
    arrays = {
        "Optimum MLL (Custom Kernel)": optMLLCust,
        "Optimum MLL (Artificial Kernel)": optMLLArtif,
        "Optimum MLL (Diagonal Kernel)": optMLLDiag,
        "RMSE (Custom Kernel)": rmseCust,
        "RMSE (Artificial Kernel)": rmseArtif,
        "RMSE (Diagonal Kernel)": rmseDiag,
        "NLPD (Custom Kernel)": nlpdCust,
        "NLPD (Artificial Kernel)": nlpdArtif,
        "NLPD (Diagonal Kernel)": nlpdDiag,
    }

    directory = join("results", NAME)
    if not exists(directory):
        os.makedirs(directory)

    # Save the arrays
    if N_REPEAT != 1:
        for name, array in arrays.items():
            np.save(join(directory, f"{name}.npy"), array)

    print("Done")


if __name__ == "__main__":
    params_global = toml.load("params.toml")["Global"]
    params_2D = toml.load("params.toml")["2D"]
    params_2D.update(params_global)
    script_2D(params_2D)
