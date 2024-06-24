from dataclasses import dataclass

import jax.numpy as jnp
from jax import grad, hessian
from jaxtyping import Array, Float

import gpjax as gpx


@dataclass
class ArtificialKernel3D(gpx.kernels.AbstractKernel):
    kernel: gpx.kernels.AbstractKernel = gpx.kernels.RBF(active_dims=[0, 1, 2])

    def __call__(
        self, X1: Float[Array, "1 D"], X2: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        
        z1 = jnp.array(X1[-1], dtype=int)
        z2 = jnp.array(X2[-1], dtype=int)

        val1 = jnp.array(X1[:-1])
        val2 = jnp.array(X2[:-1])

        def k00(val1, val2, z1, z2):
            return (z1 == z2) * self.kernel(val1, val2)

        def k11(val1, val2, z1, z2):
            hess = -jnp.array(hessian(self.kernel)(val1, val2), dtype=jnp.float64)
            diagFlag = z1 == z2
            a = (z1 + 1) % 3
            b = (z1 + 2) % 3
            return (
                diagFlag * (hess[a, a] + hess[b, b])
                - (1 - diagFlag) * hess[z1 % 3, z2 % 3]
            )

        def k01(val1, val2, z1, z2):
            grd = jnp.array(grad(self.kernel)(val1, val2))
            dvtv = (-z1 - z2) % 3
            sgn = 1 - (z1 - z2 + 1) % 3
            return sgn * grd[dvtv]

        leftFlag = z1 % 3 == z1
        upFlag = z2 % 3 == z2

        return (
            leftFlag * upFlag * k00(val1, val2, z1, z2)
            + (1 - leftFlag) * (1 - upFlag) * k11(val1, val2, z1, z2)
            + ((1 - leftFlag) * upFlag + leftFlag * (1 - upFlag))
            * k01(val1, val2, z1, z2)
        )
