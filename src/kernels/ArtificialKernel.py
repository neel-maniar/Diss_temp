from dataclasses import dataclass

import jax.numpy as jnp
from jax import grad, hessian
from jaxtyping import Array, Float

import gpjax as gpx


@dataclass
class ArtificialKernel(gpx.kernels.AbstractKernel):
    kernel: gpx.kernels.AbstractKernel = gpx.kernels.RBF(active_dims=[0, 1])

    def __call__(
        self, X1: Float[Array, "1 D"], X2: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        z1 = jnp.array(X1[-1], dtype=int)
        z2 = jnp.array(X2[-1], dtype=int)

        val1 = jnp.array(X1[:-1])
        val2 = jnp.array(X2[:-1])

        def k00(val1, val2):
            return self.kernel(val1, val2)

        def k11(val1, val2):
            hess = -jnp.array(hessian(self.kernel)(val1, val2), dtype=jnp.float64)
            k22 = hess[0, 0] + hess[1, 1]
            return k22

        def k01(val1, val2, z1, z2):
            grd = jnp.array(grad(self.kernel)(val1, val2))
            return (2 * (z1 > z2) - 1) * grd[z1 % 2 + z2 % 2]

        bottomRightFlag = (z1 == 2) * (z2 == 2)
        upperLeftFlag = (z1 % 2 == z1) * (z2 % 2 == z2)

        return bottomRightFlag * k11(val1, val2) + (
            upperLeftFlag * (z1 == z2) * k00(val1, val2)
            + (1 - bottomRightFlag) * (1 - upperLeftFlag) * k01(val1, val2, z1, z2)
        )
