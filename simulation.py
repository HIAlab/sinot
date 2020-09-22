import numpy as np
import pandas as pd


def simulate_outcome(outcome_params, data, length, dependencies, causal_effects, boarders=(0, 15)):
    """
    Simulates an outcome based on given parameters
    :param outcome_params: parameters for outcome
    :param data: already simulated data
    :param length: length of study
    :param dependencies: dependencies
    :param causal_effects: causal effects
    :param boarders: tuple of lower, upper bound
    :return: baseline_drift, underlying_state, observation
    """
    # Generate baselinedrift
    baseline_drift = gen_baseline_drift(x_0=outcome_params["X_0"], length=length, sigma=outcome_params["sigma_b"],
                                        outcome_scale=outcome_params["boarders"])

    # unterlying state
    underlying_state = baseline_drift.copy()
    for dependency in dependencies:
        underlying_state += np.array(data[dependency]) * causal_effects[
            "{} -> {}".format(dependency, outcome_params["name"])]
    underlying_state = [boarders[1] if u > boarders[1] else u for u in underlying_state]
    underlying_state = [boarders[0] if u < boarders[0] else u for u in underlying_state]

    # Observation
    observation = [round(u + np.random.normal(0, outcome_params["sigma_0"])) for u in underlying_state]

    return baseline_drift, underlying_state, observation


def simulate_treatment(study_design, days_per_period, treatment, dependencies, params, causal_effects, data):
    """
    Simulates a treatment with given parameters
    :param study_design: study design
    :param days_per_period: int, defines days per period
    :param treatment: treatment name
    :param dependencies: variables, which have an causal effect on the treatment
    :param params: additional params
    :param causal_effects: causal effects
    :param data: generated data
    :return: treatment_arr, treatment_independent, dependent_treatment_effect
    """
    treatment_arr = []
    for i in range(0, len(study_design)):
        if study_design[i] == treatment:
            treatment_arr.extend([1] * days_per_period)
        else:
            treatment_arr.extend([0] * days_per_period)
    gamma = params["gamma"]
    tau = params["tau"]
    treatment_effect = params["treatment_effect"]
    treatment_independent = gen_treatment_effect(treatment_arr, gamma, tau, treatment_effect)
    dependent_treatment_effect = treatment_independent.copy()
    for dependency in dependencies:
        dependent_treatment_effect += np.array(data[dependency]) * causal_effects[
            "{} -> {}".format(dependency, treatment)]
    return treatment_arr, treatment_independent, dependent_treatment_effect


def gen_treatment_effect(treatment, gamma, tau, treatment_effect):
    """
    Generates the treatment effect for a treatment
    :param treatment: array defines the treatment for each day
    :param gamma: integer defines the gamma in the treatment effect driver
    :param tau:  integer defines the tau in the treatment effect driver
    :param treatment_effect: treatment effect for this treatment
    :return: numpy array
    """
    x = np.empty(len(treatment))
    for time_point in range(len(treatment)):
        if time_point == 0:
            x_j = 0
        else:
            x_j = x[time_point - 1]
        x[time_point] = (x_j + ((treatment_effect - x_j) / tau) * treatment[time_point] - (x_j / gamma) * (
                1 - treatment[time_point]))
    return x


def gen_baseline_drift(x_0, length, sigma=1, outcome_scale=None):
    """
    Generates the baseline drift
    :param x_0: start baseline
    :param length: length of study
    :param sigma: sigma for normal distribution
    :param outcome_scale: defines the scale for the outcome
    :return: drift
    """
    drift = np.empty(length)
    for day in range(length):
        if day == 0:
            last_day = x_0
        else:
            last_day = drift[day - 1]
        drift_est = last_day + np.random.normal(0, sigma)
        if outcome_scale:
            drift_est = max(drift_est, outcome_scale[0])
            drift_est = min(drift_est, outcome_scale[1])
        drift[day] = drift_est
    return drift


def extract_dependencies(nodes, dependencies):
    """
    Transform dependencies from the file into a dependency dictionary
    :param nodes:
    :param dependencies:
    :return:
    """
    dependencies_dict = {}
    for node in nodes:
        dependencies_dict[node] = []
    for dependency in dependencies:
        effect_from, effect_on = dependency.split(" -> ")
        dependencies_dict[effect_on].append(effect_from)
    return dependencies_dict


def gen_normal_distribution(mean=0, std=1, **_args):
    """
    Generates a value following a normal distribution.
    :param mean:
    :param std:
    :param _args:
    :return:
    """
    return np.random.normal(mean, std)


def gen_flag(p1=0.5, **_args):
    """
    Generates a flag.
    :param p1: probability of flag = 1
    :param _args:
    :return: flag
    """
    return 1 if np.random.normal(0, 100) <= p1 * 100 else 0


def gen_unit_distribution(min_value=0, max_value=10, **_args):
    """
    Returns a value with equal probabilites
    :param min_value:
    :param max_value:
    :param _args:
    :return:
    """
    return np.random.randint(min_value, max_value)


def gen_poisson_distribution(lam, **_args):
    """
    Returns a poisson distribution based on a lam value
    :param lam:
    :param _args:
    :return:
    """
    return np.random.poisson(lam)


def gen_distribution(distribution, boarder=(0, 1), **params):
    """
    Generated a value of a given distribution and params.
    :param distribution: name of distribution (normal, poisson, unit, flag)
    :param boarder: defines the min and maximum value
    :param params:
    :return:
    """
    if distribution == "normal":
        val = gen_normal_distribution(**params)
    elif distribution == "flag":
        val = gen_flag(**params)
    elif distribution == "poisson":
        val = gen_poisson_distribution(**params)
    elif distribution == "unit":
        val = gen_unit_distribution(**params)
    else:
        val = None
    val = boarder[1] if val > boarder[1] else val
    val = boarder[0] if val < boarder[0] else val
    return val


def simulate_node(node, dependencies, length, data, params, causal_effects):
    """
    This function simulates a node / feature
    :param node: name of the feature
    :param dependencies: list of dependencies
    :param length: length of the simulation
    :param data: simulated data
    :param params: parameters for the feature
    :param causal_effects: causal effect
    :return: return a list of simulated values
    """
    result = []
    for i in range(length):
        variable = gen_distribution(**params)
        for dependency in dependencies:
            variable += data[dependency][i] * causal_effects["{} -> {}".format(dependency, node)]
        result.append(variable)
    return result


class Simulation:
    def __init__(self, parameter):
        """
        This class handles a simulation
        :param parameter: dictionary with parameters
        """

        self.exposures_params = parameter["exposures"]
        self.outcome_params = parameter["outcome"]

        #        self.length_of_study = len(study_design)*days_per_period
        self.variables = parameter["variables"]
        self.dependencies = extract_dependencies(
            [*self.variables.keys(), *self.exposures_params, self.outcome_params["name"]], parameter["dependencies"])
        self.effect_sizes = parameter["dependencies"]

    def gen_patient(self, study_design, days_per_period):
        """
        This function generates a person
        :param study_design: study design
        :param days_per_period: days per period
        :return: pandas data frame
        """
        # Define length of study
        length = len(study_design) * days_per_period

        # Generate Variables with dependencies
        def _order_nodes(dependency_dict):
            final_order = []
            dependencies = [(n, set(dependency_dict[n])) for n in dependency_dict]
            dependencies.sort(key=lambda s: len(s[1]))
            while len(dependencies) > 0:
                if len(dependencies[0][1]) == 0:
                    n = dependencies.pop(0)[0]
                    final_order.append(n)
                    for dep in dependencies:
                        dep[1].discard(n)
                else:
                    return None
            return final_order

        # Order variables based on their dependencies
        ordered_node = _order_nodes(self.dependencies)

        # Generate Data
        result = {}
        for node in ordered_node:
            # if node is exposure
            if node in list(self.exposures_params.keys()):
                t = simulate_treatment(study_design,
                                       days_per_period,
                                       node,
                                       self.dependencies[node],
                                       self.exposures_params[node],
                                       self.effect_sizes,
                                       result)
                result[node], result["{}_independent".format(node)], result["{}_dependent".format(node)] = t
            # generate outcome
            elif node == self.outcome_params["name"]:
                o = simulate_outcome(outcome_params=self.outcome_params,
                                     data=result,
                                     length=length,
                                     dependencies=self.dependencies[node],
                                     causal_effects=self.effect_sizes)
                result["baseline_drift"], result["underlying_state"], result[node] = o
            # if node is variable
            else:
                result[node] = simulate_node(node, self.dependencies[node], length, result, params=self.variables[node],
                                             causal_effects=self.effect_sizes)
        return pd.DataFrame(result)
