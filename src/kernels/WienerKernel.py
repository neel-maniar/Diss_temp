from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float

import gpjax as gpx


@dataclass
class WienerKernel(gpx.kernels.AbstractKernel):
    name: str = "Wiener"

    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        K = jnp.minimum(x, y)
        return K.squeeze()
