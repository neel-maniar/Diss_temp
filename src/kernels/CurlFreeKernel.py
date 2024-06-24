from dataclasses import dataclass

import jax.numpy as jnp
from jax import hessian
from jaxtyping import Array, Float

import gpjax as gpx


@dataclass
class CurlFreeKernel(gpx.kernels.AbstractKernel):
    kernel_g: gpx.kernels.AbstractKernel = gpx.kernels.RBF(active_dims=[0, 1, 2])

    def __call__(
        self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        z = jnp.array(X[3], dtype=int)
        zp = jnp.array(Xp[3], dtype=int)

        # convert to array to correctly index, -ve sign due to exchange symmetry (only true for stationary kernels)
        # Hessian calculated only on the first argument
        kernel_g_hessian = -jnp.array(hessian(self.kernel_g)(X, Xp), dtype=jnp.float64)[
            z
        ][zp]

        return kernel_g_hessian
