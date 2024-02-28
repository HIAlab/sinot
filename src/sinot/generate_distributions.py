### Created 28.02.2024 by thogaertner (thomas.gaertner@hpi.de)
# Contains functions for generate different distributions.

import numpy as np


def gen_normal_distribution(mean=0, std=1, **_args) -> float:
    """Generates a value from a normal disctribution

    Args:
        mean (int, optional): Expected mean. Defaults to 0.
        std (int, optional): Expected Std. Defaults to 1.

    Returns:
        float: Random Variable
    """
    return np.random.normal(mean, std)


def gen_flag(p1=0.5, **_args) -> int:
    """Generates a flag of 1 (True) or 0 (False) with a given probability.

    Args:
        p1 (float, optional): P(1). Defaults to 0.5.

    Returns:
        int: 1 or 0.
    """
    return 1 if np.random.normal(0, 100) <= p1 * 100 else 0


def gen_unit_distribution(min_value=0, max_value=10, **_args) -> float:
    """Generates a value from a unit distribution.

    Args:
        min_value (int, optional): Minimum value. Defaults to 0.
        max_value (int, optional): Maximum value. Defaults to 10.

    Returns:
        float: Random Value.
    """
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


def gen_distribution(distribution:str, boarders=(0, 1), **params)->float:
    """Generate the distribution.

    Args:
        distribution (str): Name of distribution. Valid Names: 'normal', 'flag', 'poisson', 'unit'. If none of it, it returns 'value' from params or 0.
        boarders (tuple, optional): min and max values. The random numbers will be cut off in those boarders. Defaults to (0, 1).

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
        val = gen_unit_distribution(boarders[0], boarders[1])
    else:
        val = params.get("value", 0)
    
    if boarders[1] and distribution.lower() != "not":
        val = boarders[1] if val > boarders[1] else val 
    if boarders[0] and distribution.lower() != "not":
        val = boarders[0] if val < boarders[0] else val
    return float(val)