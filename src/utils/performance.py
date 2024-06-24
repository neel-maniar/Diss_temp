"""
This module provides utility functions for performance evaluation.

Functions:
- rmse(ytest, ypred): Calculate the Root Mean Squared Error (RMSE) between the predicted values and the true values.
- nlpd(ypred, y_std, y_true): Calculate the negative log predictive density (NLPD) for a given set of predictions.
- header(string): Prints a header with a given string.
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
