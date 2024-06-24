# %%
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
    generate_harmonic_data,
    transform_data,
    add_collocation_points,
)
from utils.performance import rmse
from utils.inference import get_posterior, predict, steal_diag_params
from line_profiler import LineProfiler
from tqdm.auto import tqdm, trange
import warnings

profiler = LineProfiler()

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx


# Enable Float64 for more stable matrix inversions.
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
colors = rcParams["axes.prop_cycle"].by_key()["color"]

key = jr.key(0)

x, y, xtest, ytest = generate_harmonic_data(key, 800, 20, 1e-4)

#%%

nrRepeat = 1

if nrRepeat > 1:
    verbose = False
# log_directory = "./logs"
# if not exists(log_directory):
#     os.makedirs(log_directory)

# sys.stderr = open(join(log_directory, f"log_BFGS.txt"), "w")

# %%

errDiag_all = np.zeros((nrRepeat, 1))
errCust_all = np.zeros((nrRepeat, 1))
L_opt_diag_all = np.zeros((nrRepeat, 1))
L_opt_cust_all = np.zeros((nrRepeat, 1))

objective = gpx.objectives.ConjugateMLL(negative=True)

for i in trange(nrRepeat):
    dataset_train = gpx.Dataset(x, y)
    dataset_test = gpx.Dataset(xtest, ytest)

    # Diagonal kernel
    predictions = {}
    opt_posterior_diag, opt_MLL, diag_params = get_posterior(
        DiagonalKernel3D(),
        dataset_train,
        verbose=True,
        optimisation="scipy",
        key=master_key,
    )
    L_opt_diag_all[i] = opt_MLL.item()
    mean_diag = predict(opt_posterior_diag, dataset_test.X, dataset_train, 3)
    errDiag_all[i] = rmse(mean_diag, dataset_test.y)

    # Curl-free kernel
    print()
    opt_posterior_curlfree, opt_MLL, _ = get_posterior(
        CurlFreeKernel(),
        dataset_train,
        verbose=True,
        optimisation="scipy",
        key=master_key,
    )
    L_opt_cust_all[i] = opt_MLL.item()
    mean_curlfree = predict(opt_posterior_curlfree, dataset_test.X, dataset_train, 3)
    errCust_all[i] = rmse(mean_curlfree, dataset_test.y)

    for j, N_c in enumerate(N_c_list):
        dataset_coll_train = add_collocation_points(
            dataset_train, test_pos, N_c, master_key, 3
        )
        opt_posterior_art = steal_diag_params(
            diag_params, dataset_coll_train, ArtificialKernelExplicit3D()
        )
        opt_MLL = objective(opt_posterior_art, dataset_coll_train)
        L_opt_diagObs_all[i, j] = opt_MLL.item()
        mean_artificial = predict(
            opt_posterior_art, dataset_test.X, dataset_coll_train, 3
        )
        errDiagObs_all[i, j] = rmse(mean_artificial, dataset_test.y)

arrays = {
    "L_opt_cust_all": L_opt_cust_all,
    "L_opt_diag_all": L_opt_diag_all,
    "L_opt_diagObs_all": L_opt_diagObs_all,
    "errCust_all": errCust_all,
    "errDiag_all": errDiag_all,
    "errDiagObs_all": errDiagObs_all,
}

directory = "results/3DRefactor"

if not exists(directory):
    os.makedirs(directory)

# Save the arrays
for name, array in arrays.items():
    np.save(join(directory, f"{name}.npy"), array)
