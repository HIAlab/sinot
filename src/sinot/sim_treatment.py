### Created 28.02.2024 by thogaertner (thomas.gaertner@hpi.de)
# Contains functions for simulating the outcome

import numpy as np
from .utils import normalize

def simulate_treatment(study_design, days_per_period, treatment, dependencies, params, causal_effects, data):
    """Simulates a treatment with given parameters

    Args:
        study_design (list): List with the order of treatments.
        days_per_period (int): Length of a treatment period.
        treatment (str): Name of the treatment.
        dependencies (list): List of variables having an effect on the treatment variable.
        params (dict): parameters for the treatment such as gamma, tau and treatment effect.
        causal_effects (dict): dict specifing the effects on the treatment.
        data (dict): simulated data of the study.

    Returns:
        tuple: treatment_arr, treatment_effect
    """

    treatment_arr = []
    
    # Generate treatment array based on default study design
    for i in range(0, len(study_design)):
        if study_design[i] == treatment:
            treatment_arr.extend([1] * days_per_period)
        else:
            treatment_arr.extend([0] * days_per_period)

    # Add dependencies
    for dependency in dependencies:
        treatment_arr += normalize(data[dependency]) * causal_effects[
            "{} -> {}".format(dependency, treatment)]
    
    # Clip treatment to 1 or 0 
    treatment_arr = np.where(np.array(treatment_arr) >= 0.5, 1, 0)

    # Generate 
    treatment_effects_array = gen_treatment_effect(treatment_arr, **params)
    return treatment_arr, treatment_effects_array


def gen_treatment_effect(treatment, gamma=1, tau=1, treatment_effect=1):
    """Generate the treatment effect with wash-in and wash-out

    Args:
        treatment (list): defines the treatment for each day
        gamma (int): _description_
        tau (int): _description_
        treatment_effect (float): Effect size of the treatment

    Returns:
        np.array: values of the treatment for each day.
    """
    #Generates the treatment effect for a treatment
    #:param treatment: array defines the treatment for each day
    #:param gamma: integer defines the gamma in the treatment effect driver
    #:param tau:  integer defines the tau in the treatment effect driver
    #:param treatment_effect: treatment effect for this treatment
    #:return: numpy array
    x = np.empty(len(treatment))
    for time_point in range(len(treatment)):
        if time_point == 0:
            x_j = 0
        else:
            x_j = x[time_point - 1]
        x[time_point] = (x_j + ((treatment_effect - x_j) / tau) * treatment[time_point] - (x_j / gamma) * (
                1 - treatment[time_point]))
    return x
