### Created 28.02.2024 by thogaertner (thomas.gaertner@hpi.de)
# Contains functions for generate different distributions.

import numpy as np


def gen_normal_distribution(mu=0, sigma=1, **_args) -> float:
    """Generates a value from a normal disctribution

    Args:
        mean (int, optional): Expected mean. Defaults to 0.
        std (int, optional): Expected Std. Defaults to 1.

    Returns:
        float: Random Variable
    """
    return np.random.normal(mu, sigma)


def gen_flag(p1=0.5, **_args) -> int:
    """Generates a flag of 1 (True) or 0 (False) with a given probability.

    Args:
        p1 (float, optional): P(1). Defaults to 0.5.

    Returns:
        int: 1 or 0.
    """
    return 1 if np.random.normal(0, 100) <= p1 * 100 else 0


def gen_unit_distribution(boarders: tuple, **_args) -> float:
    """_summary_

    Args:
        boarders (tuple): _description_

    Returns:
        float: _description_
    """
    min_value, max_value = boarders
    if min_value<=max_value:
        return float(np.random.randint(min_value, max_value))
    else: 
        return float(min_value)


def gen_poisson_distribution(lam, **_args)->float:
    """Generates a value following the poisson distribution.

    Args:
        lam (int): Parameter for Poisson.

    Returns:
        float: Random value.
    """
    return float(np.random.poisson(lam=lam))


def gen_distribution(distribution:str="normal", **params)->float:
    """Generate the distribution.

    Args:
        distribution (str): Name of distribution. Valid Names: 'normal', 'flag', 'poisson', 'unit'. If none of it, it returns 'value' from params or 0.

    Returns:
        float: random value.
    """
    if distribution.lower() == "normal":
        val = gen_normal_distribution(**params)
    elif distribution.lower() == "flag":
        val = gen_flag(**params)
    elif distribution.lower() == "poisson":
        val = gen_poisson_distribution(**params)
    elif distribution.lower() == "unit":
        val = gen_unit_distribution(**params)
    else:
        val = params.get("value", 0)
    return float(val)