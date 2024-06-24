from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float

import gpjax as gpx


@dataclass
class DiagonalKernel3D(gpx.kernels.AbstractKernel):
    kernel: gpx.kernels.AbstractKernel = gpx.kernels.RBF(active_dims=[0, 1, 2])

    # kernel: gpx.kernels.AbstractKernel = gpx.kernels.RBF(active_dims=[0, 1, 2], lengthscale = jnp.array([0.115]), variance = jnp.array([0.404]))

    def __call__(
        self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:

        z = jnp.array(X[3], dtype=int)
        zp = jnp.array(Xp[3], dtype=int)

        switch = z==zp

        return switch * self.kernel(X, Xp)
