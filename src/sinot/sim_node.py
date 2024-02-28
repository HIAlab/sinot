### Created 28.02.2024 by thogaertner (thomas.gaertner@hpi.de)
# Contains functions for simulating a node

from .generate_distributions import gen_distribution

def simulate_node(node:str, dependencies:list, length:int, data:dict, params:dict, causal_effects:dict, over_time_effects:dict={}) -> list:
    """This function simulates a node / feature

    Args:
        node (str): name of the feature
        dependencies (list): list of dependencies
        length (int): length of the simulation
        data (dict): simulated data
        params (dict): parameters for the feature
        causal_effects (dict): causal effect
        over_time_effects (dict, optional): _description_. Defaults to None.

    Returns:
        list: return a list of simulated values
    """

    if params["constant"]:
        result = [gen_distribution(**params)] * length
    else:
        result = []
        for i in range(length):
            variable = gen_distribution(**params)
            for dependency in dependencies:
                variable += data[dependency][i] * causal_effects["{} -> {}".format(dependency, node)]
            if over_time_effects:
                # Iterate over all variables with an effect on out node
                for dependency in over_time_effects.keys():
                    # Iterate over all lags
                    for lag, effect in  enumerate(over_time_effects[dependency]["effects"]):
                        # Get the data of the dependeny
                        dependent_data = data.get(dependency, [])
                        # Get the value of the lagged variable, if the data is already simulated:
                        flag_lagged_value_in_study_period = (i-lag)>0
                        flag_lagged_value_is_simulated = len(dependent_data)>=(i-lag)
                        if (flag_lagged_value_in_study_period) and (flag_lagged_value_is_simulated):
                            variable +=  dependent_data[i-1-lag] * effect      
            result.append(variable)
    if params["boarders"]:
        if params["boarders"][0]:
            result = [params["boarders"][0] if val<params["boarders"][0] else val for val in result]
        if params["boarders"][1]:
            result = [params["boarders"][1] if val>params["boarders"][1] else val for val in result]
    return result
