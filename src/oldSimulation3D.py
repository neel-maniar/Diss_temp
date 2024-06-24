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
from utils.performance import rmse
from utils.oldinference import get_posterior, predict, steal_diag_params
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

mat = sio.loadmat("data/dataSet14.mat")
t = mat["data_obj"][0, 1].item()[0][:-1]
pos = mat["data_obj"][0, 1].item()[1][:-1]
mag = mat["data_obj"][0, 1].item()[2][:-1]
quat = mat["data_obj"][0, 1].item()[3][:-1]

n_train = 500
n_test = 1000
nrRepeat = 50


# log_directory = "./logs"
# if not exists(log_directory):
#     os.makedirs(log_directory)

# sys.stderr = open(join(log_directory, f"log_BFGS.txt"), "w")

# %%

N_c_list = [10, 19, 37, 72, 139, 268, 518, 1000]
# N_c_list = [10, 19]

errDiag_all = np.zeros((nrRepeat, 1))
errCust_all = np.zeros((nrRepeat, 1))
errDiagObs_all = np.zeros((nrRepeat, len(N_c_list)))
L_opt_diag_all = np.zeros((nrRepeat, 1))
L_opt_cust_all = np.zeros((nrRepeat, 1))
L_opt_diagObs_all = np.zeros((nrRepeat, len(N_c_list)))

posmag = jnp.hstack([pos, mag])

objective = gpx.objectives.ConjugateMLL(negative=True)

for i in trange(nrRepeat):
    print()
    print(i)
    master_key = jr.key(i)
    sample_pos = jr.choice(master_key, posmag, (n_train + n_test,), replace=False)
    train_posmag = sample_pos[:n_train]
    test_posmag = sample_pos[n_train:]
    train_pos = train_posmag[:, :3]
    train_mag = train_posmag[:, 3:]
    test_pos = test_posmag[:, :3]
    test_mag = test_posmag[:, 3:]

    ## Evenly spaced points
    # total_points = posmag.shape[0]
    # indices = jnp.linspace(0, total_points - 1, n_train + n_test, dtype=jnp.int32)
    # train_indices = indices[::3]
    # test_indices = jnp.setdiff1d(indices, train_indices)
    # train_pos = posmag[train_indices, :3]
    # train_mag = posmag[train_indices, 3:]
    # test_pos = posmag[test_indices, :3]
    # test_mag = posmag[test_indices, 3:]

    # Real points
    dataset_train, dataset_test = transform_data(
        train_pos, train_mag, test_pos, test_mag
    )

    # Diagonal kernel
    print("Diagonal kernel")
    predictions = {}
    opt_posterior_diag, diag_params = get_posterior(
        DiagonalKernel3D(),
        dataset_train,
        verbose=True,
        params=True,
        optimisation="scipy",
        key=master_key,
    )
    MLL = objective(opt_posterior_diag, dataset_train)
    L_opt_diag_all[i] = MLL.item()
    mean_diag = predict(opt_posterior_diag, dataset_test.X, dataset_train, 3)
    error = rmse(mean_diag, dataset_test.y)
    errDiag_all[i] = error
    print(error)

    # Curl-free kernel
    print()
    print("Curl-free kernel")
    opt_posterior_curlfree = get_posterior(
        CurlFreeKernel(),
        dataset_train,
        verbose=True,
        optimisation="scipy",
        key=master_key,
    )

    MLL = objective(opt_posterior_curlfree, dataset_train)
    L_opt_cust_all[i] = MLL.item()
    mean_curlfree = predict(opt_posterior_curlfree, dataset_test.X, dataset_train, 3)
    error = rmse(mean_curlfree, dataset_test.y)
    errCust_all[i] = error
    print(error)

    for j, N_c in enumerate(N_c_list):
        print(f"Collocation points: {N_c}")
        dataset_coll_train = add_collocation_points(
            dataset_train, test_pos, N_c, master_key, 3
        )

        print(dataset_coll_train.X.shape)
        print(dataset_coll_train.y.shape)
        opt_posterior_art = steal_diag_params(
            diag_params, dataset_coll_train, ArtificialKernelExplicit3D()
        )
        MLL = objective(opt_posterior_art, dataset_coll_train)
        L_opt_diagObs_all[i, j] = MLL.item()
        mean_artificial = predict(
            opt_posterior_art, dataset_test.X, dataset_coll_train, 3
        )
        print(mean_artificial.shape)
        print(mean_artificial)
        # mean_artificial_2 = vmap(
        #     lambda x: opt_posterior_art.predict(x, dataset_coll_train_full).mean(), 0
        # )(dataset_test.X[:, None])
        error = rmse(mean_artificial, dataset_test.y)
        errDiagObs_all[i, j] = error
        print(error)

arrays = {
    "L_opt_cust_all": L_opt_cust_all,
    "L_opt_diag_all": L_opt_diag_all,
    "L_opt_diagObs_all": L_opt_diagObs_all,
    "errCust_all": errCust_all,
    "errDiag_all": errDiag_all,
    "errDiagObs_all": errDiagObs_all,
}

directory = "results/Adam_jnp"

if not exists(directory):
    os.makedirs(directory)

# Save the arrays
for name, array in arrays.items():
    np.save(join(directory, f"{name}.npy"), array)
