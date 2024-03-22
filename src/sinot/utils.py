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


def extract_dependencies(nodes:list, dependencies:dict):
    """Transform dependencies from the file into a dependency dictionary.

     Args:
         nodes (list): List with names of nodes.
         dependencies (dict): Contains all dependencies.

    Returns:
         dict: Dict with all dependencies
    """
    dependency_dict = {n:{} for n in nodes}
    for key, value in zip(dependencies.keys(),dependencies.values()):
        if " -> " in key:
            f, t = key.split(" -> ")
            effect_size = value
            if not(t in dependency_dict.keys()):
                dependency_dict[t] = {}
            dependency_dict[t][f] = effect_size
        elif type(value)==dict:
            for f in value.keys():
                t = key
                effect_size = value[f]
                if not(t in dependency_dict.keys()):
                    dependency_dict[t] = {}
                dependency_dict[t][f] = effect_size
    return dependency_dict
