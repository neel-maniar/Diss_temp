import cola
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import config
from line_profiler import profile

config.update("jax_enable_x64", True)
import sys
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow_probability as tfp
from jaxtyping import Array, Float, install_import_hook
from matplotlib import rcParams

sys.path.append("src")
from line_profiler import LineProfiler

from kernels.ArtificialKernel3D import ArtificialKernel3D
from kernels.ArtificialKernelExplicit3D import ArtificialKernelExplicit3D
from kernels.CurlFreeKernel import CurlFreeKernel
from kernels.DiagonalKernel3D import DiagonalKernel3D
from utils.data_tools import (
    add_collocation_points,
    generate_data,
    regular_train_points,
    transform_data,
)
from utils.performance import rmse
from utils.plotting_tools import plot_data, plot_pred

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

# Enable Float64 for more stable matrix inversions.
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
colors = rcParams["axes.prop_cycle"].by_key()["color"]


n_train = 112
n_test = n_train + n_train

mat = sio.loadmat("data/dataSet14.mat")
t = mat["data_obj"][0, 1].item()[0][:-1]
pos = mat["data_obj"][0, 1].item()[1][:-1]
mag = mat["data_obj"][0, 1].item()[2][:-1]

posmag = jnp.hstack([pos, mag])
master_key = jr.key(0)

# Evenly spaced points
total_points = posmag.shape[0]
indices = jnp.linspace(0, total_points - 1, n_train + n_test, dtype=jnp.int32)
train_indices = indices[::3]
test_indices = jnp.setdiff1d(indices, train_indices)
train_pos = posmag[train_indices, :3]
train_mag = posmag[train_indices, 3:]
test_pos = posmag[test_indices, :3]
test_mag = posmag[test_indices, 3:]

dataset_train, dataset_test = transform_data(train_pos, train_mag, test_pos, test_mag)
dataset_coll_train_full = add_collocation_points(
    dataset_train, test_pos, n_test, master_key, 3
)

x = dataset_coll_train_full.X
t = dataset_test.X


def setup(t, x):
    kernel = ArtificialKernelExplicit3D()

    Kxx = kernel.gram(x)
    Kxx += cola.ops.I_like(Kxx) * 1e-6
    # Σ = Kxx + Io²
    Sigma = Kxx + cola.ops.I_like(Kxx) * 1e-6
    Sigma = cola.PSD(Sigma)

    Kxt = kernel.cross_covariance(x, t)
    return Sigma, Kxt


@profile
def solve_np(Sigma, Kxt):
    Sigma = Sigma.to_dense()
    return np.linalg.solve(Sigma, Kxt)


@profile
def solve_cola(Sigma, Kxt):
    return cola.solve(Sigma, Kxt)


@profile
def main(t, x):
    Sigma, Kxt = setup(t, x)
    print("Sigma matrix summary:")
    print("Sigma type", type(Sigma))
    print("Sigma mean", Sigma.to_dense().mean())
    print("Sigma std", Sigma.to_dense().std())
    print("Sigma determinant", cola.linalg.logdet(Sigma))
    print(f"Sigma.shape = {Sigma.shape}")
    print()
    print("Kxt matrix summary:")
    print("Kxt type", type(Kxt))
    print("Kxt mean", Kxt.mean())
    print("Kxt std", Kxt.std())
    print(f"Kxt.shape = {Kxt.shape}")
    print(f"n_train = {n_train}, n_test = {n_test}")

    print("Numpy solving")
    a = solve_np(Sigma, Kxt)

    print("Cola solving")
    b = solve_cola(Sigma, Kxt)

    print("Difference between solutions", jnp.linalg.norm(a - b))

    Sigma = Sigma.to_dense()

    print(
        "Difference between reconstruction and original (Numpy)",
        jnp.linalg.norm(jnp.matmul(Sigma, a) - Kxt),
    )
    print(
        "Difference between reconstruction and original (Cola)",
        jnp.linalg.norm(jnp.matmul(Sigma, b) - Kxt),
    )

    print("Done")


if __name__ == "__main__":
    main(t, x)
