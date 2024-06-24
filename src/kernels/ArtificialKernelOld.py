from dataclasses import dataclass

import jax.numpy as jnp
from jax import grad, hessian
from jaxtyping import Array, Float

import gpjax as gpx


@dataclass
class ArtificialKernelOld(gpx.kernels.AbstractKernel):
    kernel: gpx.kernels.AbstractKernel = gpx.kernels.RBF(active_dims=[0, 1])

    def __call__(
        self, X1: Float[Array, "1 D"], X2: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        z1 = jnp.array(X1[-1], dtype=int)
        z2 = jnp.array(X2[-1], dtype=int)

        k20ork21 = -jnp.abs(z1 - z2) + 2  # 0 if z1=2,z2=0, 1 if z1=2,z2=1
        k22_switch = jnp.heaviside(z1 + z2 - 4, 1)
        k00_switch = z1 == z2
        k2021_switch = jnp.heaviside(z1 * z2 + z1 + z2 - 1, 0)

        upper = jnp.sign(z1 - z2)

        val1 = jnp.array(X1[:-1])
        val2 = jnp.array(X2[:-1])

        k00 = self.kernel(val1, val2)  # could change to X1,X2 if needed.

        grd = jnp.array(grad(self.kernel)(val1, val2))

        k2021 = grd[k20ork21]

        hess = -jnp.array(hessian(self.kernel)(val1, val2), dtype=jnp.float64)

        k22 = hess[0, 0] + hess[1, 1]

        # k00   if z1=0,   z2=0, or z1=1, z2=1
        # 0     if z1=0,   z2=1, or z1=1, z2=0
        # k2021 if z1=0,1, z2=2, or z1=2, z2=0,1
        # k22   if z1=2,   z2=2

        return k22_switch * k22 + (1 - k22_switch) * (
            k00_switch * k00 + (1 - k00_switch) * (k2021_switch * upper * k2021)
        )
