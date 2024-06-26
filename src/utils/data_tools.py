"""
Data tools
=====================

This module provides utility functions for generating and transforming data for a toy problem.

"""

import jax.numpy as jnp
import jax.random as jr
import gpjax as gpx

from typing import List, Tuple, Union, TypeVar

# Rest of the code...
"""

"""

import jax.numpy as jnp
import jax.random as jr
import gpjax as gpx

from typing import List, Tuple, Union, TypeVar


def f1(x1, x2, a=0.01):
    """
    Compute the value of f1 function.

    Parameters:
    -----------
    x1 : float
        The first input value.
    x2 : float
        The second input value.
    a : float, optional
        The coefficient value (default is 0.01).

    Returns:
    --------
    float
        The computed value of the f1 function.
    """
    return jnp.exp(-a * x1 * x2) * (a * x1 * jnp.sin(x1 * x2) - x1 * jnp.cos(x1 * x2))


def f2(x1, x2, a=0.01):
    """
    Compute the value of f2 function.

    Parameters:
    -----------
    x1 : float
        The first input value.
    x2 : float
        The second input value.
    a : float, optional
        The coefficient value (default is 0.01).

    Returns:
    --------
    float
        The computed value of the f2 function.
    """
    return jnp.exp(-a * x1 * x2) * (x2 * jnp.cos(x1 * x2) - a * x2 * jnp.sin(x1 * x2))


def f(x, a=0.01):
    return jnp.transpose(jnp.array([f1(x[:, 0], x[:, 1], a), f2(x[:, 0], x[:, 1], a)]))


def generate_data(key, a=0.01, n=50, n_test_1d=20, noise=10**-4):
    """
    Generate data for the toy problem.

    Parameters
    ----------
    key : jax.random.prngkey
        The random key used for generating the data.
    a : float, optional
        A parameter in the toy latent function, by default 0.01.
    n : int, optional
        Number of training points, by default 50.
    n_test_1d : int, optional
        Number of test points in 1D, by default 20.
    noise : float, optional
        Noise level of the generated training data, by default 10**-4.

    Returns
    -------
    x : jnp.ndarray
        Training input data of shape (n, 2).
    y : jnp.ndarray
        Training output data of shape (n, 2).
    xtest : jnp.ndarray
        Test input data of shape (n_p, 2).
    ytest : jnp.ndarray
        Test output data of shape (n_p, 2).
    """

    # get two keys
    key, subkey = jr.split(key)

    # get the training input data
    x = jr.uniform(key=key, minval=0.0, maxval=4.0, shape=(n, 2)).reshape(-1, 2)

    # noiseless output data
    signal = f(x, a)
    # add noise to the output data
    y = signal + jr.normal(subkey, shape=signal.shape) * noise

    # get the test input data
    x1_test = jnp.linspace(0, 4, n_test_1d)
    x2_test = jnp.linspace(0, 4, n_test_1d)
    xtest = jnp.vstack(jnp.dstack(jnp.meshgrid(x1_test, x2_test)))

    # get the test output data
    ytest = f(xtest)
    return x, y, xtest, ytest


def regular_train_points(key, a=0.01, n=50, noise=10**-4):
    """
    Generate regular training points with noise.

    Parameters
    ----------
    key : jax.random.PRNGKey
        The random key for generating noise.
    a : float, optional
        The parameter for the function f(x, a). Default is 0.01.
    n : int, optional
        The number of training points. Default is 50.
    noise : float, optional
        The standard deviation of the noise added to the output data. Default is 10**-4.

    Returns
    -------
    x : jax.numpy.ndarray
        The input training points of shape (n, 2).
    y : jax.numpy.ndarray
        The output training points of shape (n,).

    """
    n_1D = int(jnp.sqrt(n))
    x = jnp.linspace(0, 4, n_1D, dtype=jnp.float64)
    X, Y = jnp.meshgrid(x, x)
    x = jnp.vstack((Y.flatten(), X.flatten())).T
    # Noiseless output data
    signal = f(x, a)
    # Add noise to the output data
    y = signal + jr.normal(key, shape=signal.shape) * noise
    return x, y


def label_observations(data: jnp.ndarray, output_dim: int) -> jnp.ndarray:
    """
    Label the observations in the input data with 0 or 1 depending on the dimension of the output.

    Parameters
    ----------
        data (jnp.ndarray): The input data (x).
        output_dim (int): The dimension of the output.

    Returns
    -------
        jnp.ndarray: The labeled data, where each x_i is repeated twice, first labeled by 0s and then labeled by 1s.
    """
    label = jnp.repeat(jnp.arange(output_dim), data.shape[0]).reshape(-1, 1)
    x_repeat = jnp.tile(data, (output_dim, 1))
    return jnp.hstack((x_repeat, label))


def transform_data(x, y, xtest, ytest):
    """
    Transform the input data and labels into a dataset for training and testing.

    Parameters
    ----------
    x : numpy.ndarray
        Input data for training.
    y : numpy.ndarray
        Labels for training.
    xtest : numpy.ndarray
        Input data for testing.
    ytest : numpy.ndarray
        Labels for testing.

    Returns
    -------
    dataset_train : gpx.Dataset
        Dataset object containing the transformed training data and labels.
    dataset_test : gpx.Dataset
        Dataset object containing the transformed testing data and labels.
    """
    # flatten out and label train points
    assert x.shape[0] == y.shape[0], "Different number of training inputs and outputs"
    assert (
        xtest.shape[0] == ytest.shape[0]
    ), "Different number of test inputs and outputs"
    output_dim = y.shape[1]
    real_observations = label_observations(x, output_dim)
    y_1d = y.reshape((-1, 1), order="F")

    # flatten out and label test points
    real_test_points = label_observations(xtest, output_dim)
    ytest_1d = ytest.reshape((-1, 1), order="F")
    dataset_test = gpx.Dataset(real_test_points, ytest_1d)
    dataset_train = gpx.Dataset(real_observations, y_1d)
    return dataset_train, dataset_test


def add_collocation_points(dataset_train, xtest, num_coll, key, functional_dim=1, regular=False):
    """
    Add collocation points to the dataset.

    Parameters
    ----------
    - dataset_train (gpx.Dataset): Training dataset.
    - xtest (ndarray, N by D): Test points.
    - num_coll (int): Number of collocation points to add.
    - key (jax.random.PRNGKey): Random key for generating collocation points.
    - new_dim (int, optional): Number of additional dimensions to add to the collocation points. Defaults to 1.

    Returns
    -------
    - gpx.Dataset: Updated dataset with collocation points added.
    """
    if num_coll > xtest.shape[0]:
        raise ValueError(
            "Number of collocation points should be less than or equal to the number of test points."
        )
    output_dim = dataset_train.X[-1, -1]
    zeros = jnp.zeros(num_coll * functional_dim).reshape(-1, 1)
    y_output = jnp.vstack((dataset_train.y, zeros))
    if regular:
        indices = jnp.linspace(0, xtest.shape[0] - 1, num_coll, dtype=jnp.int32)
        r = xtest[indices]
    else:
        r = jr.choice(key, xtest, (num_coll,), replace=False)
    extra_dims = jnp.arange(output_dim + 1, output_dim + 1 + functional_dim)
    repeated_extra_dims = jnp.repeat(extra_dims, num_coll)
    column = repeated_extra_dims.reshape(-1, 1)
    artif_obs = jnp.hstack((jnp.repeat(r, functional_dim, axis=0), column))
    return gpx.Dataset(jnp.vstack((dataset_train.X, artif_obs)), y_output)
