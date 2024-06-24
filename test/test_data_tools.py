import sys

import jax.numpy as jnp
import jax.random as jr
from jax import config

config.update("jax_enable_x64", True)
sys.path.append("src")

from utils import data_tools

x = jnp.array([[0, 1], [2, 3]], dtype=jnp.float64)
y = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.float64)
xtest = jnp.array([[8, 9]], dtype=jnp.float64)
ytest = jnp.array([[12, 13, 14]], dtype=jnp.float64)


def test_transform_data():
    data_train, data_test = data_tools.transform_data(x, y, xtest, ytest)

    # Dimensions
    N = x.shape[0]
    D = x.shape[1]
    K = y.shape[1]

    assert jnp.allclose(
        data_train.X,
        jnp.array(
            [
                [0, 1, 0],
                [2, 3, 0],
                [0, 1, 1],
                [2, 3, 1],
                [0, 1, 2],
                [2, 3, 2],
            ]
        ),
    )

    assert jnp.allclose(
        data_train.y, jnp.array([[0.0], [3.0], [1.0], [4.0], [2.0], [5.0]])
    )

    assert jnp.allclose(
        data_test.X, jnp.array([[8.0, 9.0, 0.0], [8.0, 9.0, 1.0], [8.0, 9.0, 2.0]])
    )

    assert jnp.allclose(data_test.y, jnp.array([[12.0], [13.0], [14.0]]))


def test_add_collocation_points():
    dataset_train, _ = data_tools.transform_data(x, y, xtest, ytest)
    key = jr.key(0)
    dataset_train_1 = data_tools.add_collocation_points(dataset_train, xtest, 1, key)
    assert jnp.allclose(dataset_train_1.X[-1], jnp.array([8.0, 9.0, 3.0]))
    assert jnp.allclose(dataset_train_1.y[-1], jnp.array([0.0]))

    dataset_train_2 = data_tools.add_collocation_points(dataset_train, xtest, 1, key, 3)
    assert jnp.allclose(dataset_train_2.X[-1], jnp.array([8.0, 9.0, 5.0]))
    assert jnp.allclose(dataset_train_2.y[-1], jnp.array([0.0]))


test_transform_data()
test_add_collocation_points()
