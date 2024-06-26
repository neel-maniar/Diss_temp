"""
This module provides utility functions for performance evaluation.

If run directly, it gives some benchmark RMSE and NLPD values for zero and mean predictions. 
"""

import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax as tfp_jax


def rmse(ytest, ypred):
    """
    Calculate the Root Mean Squared Error (RMSE) between the predicted values and the true values.

    Parameters
    ----------
    - ytest (numpy.ndarray):
        The true values.
    - ypred (numpy.ndarray):
        The predicted values.

    Returns
    -------
    - float:
        The RMSE value.

    """
    ytest = ytest.flatten("F")
    ypred = ypred.flatten("F")
    N = ytest.shape[0]
    return jnp.sqrt(jnp.sum((ypred - ytest) ** 2) / N)


def nlpd(ypred, y_std, y_true):
    """
    Calculate the negative log predictive density (NLPD) for a given set of predictions.

    Parameters
    ----------
    - ypred (array-like):
        Predicted values.
    - y_std (array-like):
        Standard deviation of the predicted values.
    - y_true (array-like):
        True values.

    Returns
    -------
    - float:
        The negative log predictive density.

    """
    test_grid = y_true.flatten("F")
    ypred = ypred.flatten("F")
    y_std = y_std.flatten("F")
    normal = tfp_jax.distributions.Normal(loc=ypred, scale=y_std)
    return -jnp.sum(normal.log_prob(test_grid))


def header(string: str):
    """
    Prints a header with a given string.

    Parameters
    ----------
    string (str):
        The string to be printed as the header.

    """
    print(f"\n{string}\n{'-'*len(string)}")


if __name__ == "__main__":
    # 2D
    import toml
    import sys
    import jax

    jax.config.update("jax_enable_x64", True)

    sys.path.append("src")

    from utils.data_tools import generate_data, transform_data

    params_global = toml.load("params.toml")["Global"]
    params = toml.load("params.toml")["2D"]
    params.update(params_global)

    A = params["a"]
    NOISE = params["noise"]
    N_TRAIN = params["N_train"]
    N_TEST_1D = params["N_test_1D"]
    master_key = jr.PRNGKey(0)
    x, y, xtest, ytest = generate_data(master_key, A, N_TRAIN, N_TEST_1D, NOISE)

    zeros = jnp.zeros_like(ytest)
    ones = jnp.ones_like(ytest)
    print("RMSE for a zero prediction (2D)", rmse(zeros, ytest))
    print(
        "NLPD for a Normal(0,1) prediction (2D)",
        nlpd(zeros, ones, ytest),
    )

    # 3D
    import scipy.io as sio

    params_3D = toml.load("params.toml")["3D"]
    params.update(params_3D)
    N_TRAIN = params["N_train"]
    N_TEST = params["N_test"]

    mat = sio.loadmat("data/dataSet14.mat")
    pos = mat["data_obj"][0, 1].item()[1][:-1]
    mag = mat["data_obj"][0, 1].item()[2][:-1]

    posmag = jnp.hstack([pos, mag])
    total_points = posmag.shape[0]
    indices = jnp.linspace(0, total_points - 1, N_TRAIN + N_TEST, dtype=jnp.int32)
    train_indices = indices[::3]
    test_indices = jnp.setdiff1d(indices, train_indices)
    test_mag = posmag[test_indices, 3:]

    zeros = jnp.zeros_like(test_mag)
    ones = jnp.ones_like(test_mag)
    mean = jnp.mean(test_mag) + zeros
    print("RMSE for a zero prediction (3D)", rmse(zeros, test_mag))
    print("NLPD for a Normal(0,1) prediction (3D)", nlpd(zeros, ones, test_mag))
    print("RMSE for a mean prediction (3D)", rmse(mean, test_mag))
    print("NLPD for a Normal(mean,1) prediction (3D)", nlpd(mean, ones, test_mag))
