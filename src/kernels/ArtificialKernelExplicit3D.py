from dataclasses import dataclass

from beartype.typing import Union
import jax.numpy as jnp
from jaxtyping import Float
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from gpjax.base import param_field
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.stationary.utils import squared_distance
from gpjax.typing import (
    Array,
    ScalarFloat,
)


@dataclass
class ArtificialKernelExplicit3D(AbstractKernel):
    lengthscale: Union[ScalarFloat, Float[Array, " D"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus()
    )
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    name: str = "ArtificialKernelExplicit3D"

    def __call__(
        self, X1: Float[Array, "1 D"], X2: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:

        z1 = jnp.array(X1[-1], dtype=int)
        z2 = jnp.array(X2[-1], dtype=int)

        val1 = jnp.array(X1[:-1])
        val2 = jnp.array(X2[:-1])

        x = val1 / self.lengthscale
        y = val2 / self.lengthscale
        K = self.variance * jnp.exp(-0.5 * squared_distance(x, y))

        def k00(z1, z2):
            return (z1 == z2) * K

        def k11(val1, val2, z1, z2):
            diagFlag = z1 == z2
            a = (z1 + 1) % 3
            b = (z1 + 2) % 3
            z1 = z1 % 3
            z2 = z2 % 3
            return K * (
                diagFlag
                * (2 / self.lengthscale**2 - (x[a] - y[a]) ** 2 - (x[b] - y[b]) ** 2)
                + (1 - diagFlag) * (x[z1] - y[z1]) * (x[z2] - y[z2])
            )

        def k01(z1, z2):
            dvtv = (-z1 - z2) % 3
            sgn = 1 - (z1 - z2 + 1) % 3
            grd = K * (y[dvtv] - x[dvtv])
            return sgn * grd

        leftFlag = z1 % 3 == z1
        upFlag = z2 % 3 == z2

        return (
            leftFlag * upFlag * k00(z1, z2)
            + (1 - leftFlag) * (1 - upFlag) * k11(val1, val2, z1, z2)
            + ((1 - leftFlag) * upFlag + leftFlag * (1 - upFlag)) * k01(z1, z2)
        )
