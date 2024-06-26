from jax import config

config.update("jax_enable_x64", True)
import sys
import jax.numpy as jnp
import numpy as np
import jax.random as jr
import matplotlib.pyplot as plt
from jaxtyping import Array, Float, install_import_hook
from matplotlib import rcParams
import scipy.io as sio
import os
from os.path import exists, join

sys.path.append("src")
from kernels.DiagonalKernel3D import DiagonalKernel3D
from kernels.CurlFreeKernel import CurlFreeKernel
from kernels.ArtificialKernelExplicit3D import ArtificialKernelExplicit3D
from utils.data_tools import (
    transform_data,
    add_collocation_points,
)
from utils.performance import rmse, nlpd, header
from utils.inference import (
    predict,
    steal_diag_params,
    initialise_gp,
    optimise_mll,
)
from utils.plotting_tools import plot_3D_data, plot_3D_pred
from tqdm.auto import tqdm, trange
import warnings
from jax.flatten_util import ravel_pytree

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx


# %%


zero_mean = gpx.mean_functions.Zero()


def script_3D(params):
    mat = sio.loadmat("data/dataSet14.mat")
    pos = mat["data_obj"][0, 1].item()[1][:-1]
    mag = mat["data_obj"][0, 1].item()[2][:-1]
    # NOISE = params["noise"]
    N_TRAIN = params["N_train"]
    N_TEST = params["N_test"]
    N_C_LIST = params["N_c_list"]
    N_REPEAT = params["nrRepeat"]
    REGULAR = params["regular"]
    OPTIMISER = params["optimiser"].lower()
    PLOT = params["plot"]
    NOISE = params["noise"]
    NAME = params["name"]

    if PLOT:
        magnitude = np.linalg.norm(mag, axis=1)
        plot_3D_data(pos, magnitude)

    initial_lengthscale = 0.5
    initial_variance = 0.5

    initial_params = jnp.array(
        [2e-2, initial_lengthscale, initial_variance], dtype=jnp.float64
    )

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

    if N_REPEAT == 1:
        verbose = True
    else:
        verbose = False

    posmag = jnp.hstack([pos, mag])

    objective = gpx.objectives.ConjugateMLL(negative=True)

    for i in trange(N_REPEAT):
        master_key = jr.key(i)

        if not REGULAR:
            sample_pos = jr.choice(
                master_key, posmag, (N_TRAIN + N_TEST,), replace=False
            )
            train_posmag = sample_pos[:N_TRAIN]
            test_posmag = sample_pos[N_TRAIN:]
            train_pos = train_posmag[:, :3]
            train_mag = train_posmag[:, 3:]
            test_pos = test_posmag[:, :3]
            test_mag = test_posmag[:, 3:]
            train_mag = train_mag + NOISE * jr.normal(master_key, train_mag.shape)
        else:
            total_points = posmag.shape[0]
            indices = jnp.linspace(
                0, total_points - 1, N_TRAIN + N_TEST, dtype=jnp.int32
            )
            train_indices = indices[::3]
            test_indices = jnp.setdiff1d(indices, train_indices)
            train_pos = posmag[train_indices, :3]
            train_mag = posmag[train_indices, 3:]
            test_pos = posmag[test_indices, :3]
            test_mag = posmag[test_indices, 3:]

        # Real points
        dataset_train, dataset_test = transform_data(
            train_pos, train_mag, test_pos, test_mag
        )

        # Diagonal kernel
        kernel_name = "Diagonal Kernel"
        if verbose:
            header(kernel_name)

        posterior = initialise_gp(
            DiagonalKernel3D, zero_mean, dataset_train, 3, initial_params
        )
        opt_posterior, opt_MLL = optimise_mll(
            posterior,
            dataset_train,
            verbose=verbose,
            optimisation=OPTIMISER,
            key=master_key,
            NIters=500,
        )
        optMLLDiag[i] = opt_MLL.item()
        diag_params = ravel_pytree(opt_posterior)[0]
        ypred, y_std = predict(opt_posterior, dataset_test.X, dataset_train)
        rmseDiag[i] = rmse(ypred, dataset_test.y)
        nlpdDiag[i] = nlpd(ypred, y_std, dataset_test.y)
        if verbose:
            print(f"{kernel_name} rmse: {rmseDiag[i].item()}")
        if PLOT:
            plot_3D_pred(
                dataset_train, dataset_test, ypred, f"{NAME}_{kernel_name}", kernel_name
            )

        # Curl-free kernel
        kernel_name = "Curl-free Kernel"
        if verbose:
            header(kernel_name)

        posterior = initialise_gp(
            CurlFreeKernel, zero_mean, dataset_train, 3, initial_params
        )
        opt_posterior, opt_MLL = optimise_mll(
            posterior,
            dataset_train,
            verbose=verbose,
            optimisation=OPTIMISER,
            key=master_key,
            NIters=500,
        )
        optMLLCust[i] = opt_MLL.item()
        ypred, y_std = predict(opt_posterior, dataset_test.X, dataset_train)
        rmseCust[i] = rmse(ypred, dataset_test.y)
        nlpdCust[i] = nlpd(ypred, y_std, dataset_test.y)
        if verbose:
            print(f"{kernel_name} rmse: {rmseCust[i].item()}")
        if PLOT:
            plot_3D_pred(
                dataset_train, dataset_test, ypred, f"{NAME}_{kernel_name}", kernel_name
            )

        for j, N_c in enumerate(N_C_LIST):
            kernel_name = f"Artificial Kernel, $N_c$ = {N_c}"
            if verbose:
                header(kernel_name)
            dataset_coll_train = add_collocation_points(
                dataset_train, test_pos, N_c, master_key, 3
            )
            opt_posterior_art = steal_diag_params(
                diag_params, dataset_coll_train, ArtificialKernelExplicit3D
            )
            opt_MLL = objective(opt_posterior_art, dataset_coll_train)
            optMLLArtif[i, j] = opt_MLL.item()
            ypred, y_std = predict(
                opt_posterior_art, dataset_test.X, dataset_coll_train
            )
            rmseArtif[i, j] = rmse(ypred, dataset_test.y)
            nlpdArtif[i, j] = nlpd(ypred, y_std, dataset_test.y)
            if verbose:
                print(f"{kernel_name} rmse: {rmseArtif[i,j].item()}")
            if PLOT:
                plot_3D_pred(
                    dataset_train,
                    dataset_test,
                    ypred,
                    f"{NAME}_{kernel_name}",
                    kernel_name,
                )

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

    directory = f"results/{NAME}"

    if not exists(directory):
        os.makedirs(directory)

    # Save the arrays
    for array_name, array in arrays.items():
        np.save(join(directory, f"{array_name}.npy"), array)


# %%
if __name__ == "__main__":
    import toml

    params = toml.load("params.toml")
    global_params = toml.load("params.toml")["Global"]
    params_3D = toml.load("params.toml")["3D"]
    params_3D.update(global_params)
    script_3D(params_3D)
