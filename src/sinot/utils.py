### Created 28.02.2024 by thogaertner (thomas.gaertner@hpi.de)
# Contains helper functions

import numpy as np
import pandas as pd

def normalize(data):
    """Normalize given data with mean=0 and std=1

    Args:
        data (list): Original data.

    Returns:
        np.array: normalized data.
    """
    data = np.array(data)
    return (data - data.mean())/data.std()


def extract_dependencies(nodes, dependencies):
    """Transform dependencies from the file into a dependency dictionary.

    Args:
        nodes (list): List with names of nodes.
        dependencies (list, dict): Contains all dependencies.

    Returns:
        dict: Dict with all dependencies
    """
    dependencies_dict = {}
    for node in nodes:
        dependencies_dict[node] = []
    for dependency in dependencies:
        effect_from, effect_on = dependency.split(" -> ")
        dependencies_dict[effect_on].append(effect_from)
    return dependencies_dict

