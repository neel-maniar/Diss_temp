import cola
import cola.linalg
import jax.numpy as jnp
import jax.random as jr
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

dim1 = 1001
dim2 = 500
key = jr.key(0)

A = jr.normal(key, (dim1, dim1))
Sigma = A @ A.T  # Ensure Sigma is PSD
Sigma += 0.1 * jnp.eye(dim1)  # Ensure Sigma has positive determinant
Sigma = cola.ops.Dense(Sigma)  # Convert Sigma to a cola Dense object
# Sigma = cola.PSD(Sigma)  # Tell cola that Sigma is PSD

Kxt = jr.normal(key, (dim1, dim2))

print(f"Sigma signed log determinant (numpy) {jnp.linalg.slogdet(Sigma.to_dense())}\n")

jnp_solution = jnp.linalg.solve(Sigma.to_dense(), Kxt)
cola_solution = cola.solve(Sigma, Kxt)

print("Difference between solutions", jnp.linalg.norm(jnp_solution - cola_solution))

Sigma = Sigma.to_dense()

print(
    "Difference between reconstruction and original (Numpy)",
    jnp.linalg.norm(jnp.matmul(Sigma, jnp_solution) - Kxt),
)

print(
    "Difference between reconstruction and original (Cola)",
    jnp.linalg.norm(jnp.matmul(Sigma, cola_solution) - Kxt),
)
