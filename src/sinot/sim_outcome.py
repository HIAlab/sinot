### Created 28.02.2024 by thogaertner (thomas.gaertner@hpi.de)
# Contains functions for simulating the outcome

import numpy as np

def simulate_outcome(outcome_params, data, length, dependencies, causal_effects, treatments, boarders=(0, 15), over_time_effects={}):
    """Simulates an outcome based on given parameters

    Args:
        outcome_params (dict): parameters for outcome
        data (dict): Dictonary with simulated data
        length (int): length of study.
        dependencies (list): list with names of node having an effect on the outcome.
        causal_effects (dict): dict with a list of causal effects and their (linear) effect size.
        treatments (list): list of treatments
        boarders (tuple, optional): tuple of (lower, upper) bound. Defaults to (0, 15).
        over_time_effects (dict, optional): Specified over time dependencies. Defaults to None.

    Returns:
        tuple: baseline_drift, underlying_state, observation
    """
    
    # Generate baseline drift
    baseline_drift = gen_baseline_drift(x_0=outcome_params["X_0"], length=length, sigma=outcome_params["sigma_b"],
                                        outcome_scale=outcome_params["boarders"], mu=outcome_params.get("mu_b",0))

    # underlying state
    underlying_state = baseline_drift.copy()
    for dependency in dependencies:
        if dependency in treatments:
            underlying_state += np.array(data["{}_effect".format(dependency)])
        else:
            underlying_state += np.array(data[dependency]) * causal_effects[
                "{} -> {}".format(dependency, outcome_params["name"])]

    # TODO: Needs to be double checked
    if over_time_effects:
        for dependency in list(over_time_effects):
            for i in range(len(underlying_state)):
                for lag, effect in  enumerate(over_time_effects[dependency]["effects"]):
                    underlying_state[i] += data[dependency][i-1-lag] * effect if (i-1-lag)>=0 else 0 

    underlying_state = [boarders[1] if u > boarders[1] else u for u in underlying_state]
    underlying_state = [boarders[0] if u < boarders[0] else u for u in underlying_state]

    # Observation
    observation = [round(u + np.random.normal(0, outcome_params["sigma_0"])) for u in underlying_state]

    observation = [boarders[1] if u > boarders[1] else u for u in observation]
    observation = [boarders[0] if u < boarders[0] else u for u in observation]

    return baseline_drift, underlying_state, observation



def gen_baseline_drift(x_0, length, sigma=1, mu=0, outcome_scale=None) -> np.array:
    """This function generates a baseline drift of 'length' starting with 'x_0'. The baselinedrift follows the wiener process.

    Args:
        x_0 (int): Start value.
        length (int): Length of the simulation.
        sigma (int, optional): Sigma for noise. Defaults to 1.
        mu (int, optional): Mean of the noise. Positive values indicating an expected increase of the baseline over time, negative a decrease. Defaults to 0.
        outcome_scale (tuple, optional): Boarders for the outcome value. Defaults to None.

    Returns:
        np.array: Values of the baselinedrift.
    """
    #Generates the baseline drift
    #:param x_0: start baseline
    #:param length: length of study
    #:param sigma: sigma for normal distribution
    #:param outcome_scale: defines the scale for the outcome
    #:return: drift
    drift = np.empty(length)
    for day in range(length):
        if day == 0:
            last_day = x_0
        else:
            last_day = drift[day - 1]
        drift_est = last_day + np.random.normal(mu, sigma)
        if outcome_scale:
            drift_est = max(drift_est, outcome_scale[0])
            drift_est = min(drift_est, outcome_scale[1])
        drift[day] = drift_est
    return drift
