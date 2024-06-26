import cola
import cola.linalg
import jax.numpy as jnp
import jax.random as jr
from jax import config
import jax

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

dim = 1000
master_key = jr.key(0)

A = jr.normal(master_key, (dim, dim))
Sigma = A @ A.T  # Ensure Sigma is PSD
Sigma += 0.1 * jnp.eye(dim)  # Ensure Sigma has strictly positive determinant
Sigma = cola.ops.Dense(Sigma)  # Convert Sigma to a cola Dense object
Sigma = cola.PSD(Sigma)  # Tell cola that Sigma is PSD

# print(cola.__version__)
# print(jax.__version__)

# print("Signed Log Determinant:")
# print("-----------------------")
# print("(jax.numpy)", jnp.linalg.slogdet(Sigma.to_dense()))
# print("(cola)", cola.linalg.slogdet(Sigma))
