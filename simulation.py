import numpy as np
import pandas as pd


def simulate_outcome(outcome_params, data, length, dependencies, causal_effects, treatments, boarders=(0, 15), over_time_effects=None):
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
    # Generate baseline drift
    baseline_drift = gen_baseline_drift(x_0=outcome_params["X_0"], length=length, sigma=outcome_params["sigma_b"],
                                        outcome_scale=outcome_params["boarders"])

    # underlying state
    underlying_state = baseline_drift.copy()
    for dependency in dependencies:
        if dependency in treatments:
            underlying_state += np.array(data["{}_effect".format(dependency)])
        else:
            underlying_state += np.array(data[dependency]) * causal_effects[
                "{} -> {}".format(dependency, outcome_params["name"])]

    if over_time_effects:
        for dependency in list(over_time_effects):
            for i in len(underlying_state):
                for lag, effect in  enumerate(over_time_effects[dependency]["effects"]):
                    underlying_state[i] += data[dependency][i-1-lag] * effect if (i-1-lag)>=0 else 0 

    underlying_state = [boarders[1] if u > boarders[1] else u for u in underlying_state]
    underlying_state = [boarders[0] if u < boarders[0] else u for u in underlying_state]

    # Observation
    observation = [round(u + np.random.normal(0, outcome_params["sigma_0"])) for u in underlying_state]

    observation = [boarders[1] if u > boarders[1] else u for u in observation]
    observation = [boarders[0] if u < boarders[0] else u for u in observation]

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
    :return: treatment_arr, treatment_effect
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
        treatment_arr += np.array(data[dependency]) * causal_effects[
            "{} -> {}".format(dependency, treatment)]
    
    # Clip treatment to 1 or 0 
    treatment_arr = np.where(np.array(treatment_arr)>= 0.5, 1, 0)

    # Get Params
    gamma = params["gamma"]
    tau = params["tau"]
    treatment_effect = params["treatment_effect"]
    
    # Generate 
    treatment_effects_array = gen_treatment_effect(treatment_arr, gamma, tau, treatment_effect)
    return treatment_arr, treatment_effects_array


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
    Returns a value with equal probabilities
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


def gen_distribution(distribution, boarders=(0, 1), **params):
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
        val = gen_unit_distribution(boarders[0], boarders[1])
    else:
        val = None
    
    if boarders[1]:
        val = boarders[1] if val > boarders[1] else val 
    if boarders[0]:
        val = boarders[0] if val < boarders[0] else val
    return val


def simulate_node(node, dependencies, length, data, params, causal_effects, over_time_effects=None):
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

    if params["constant"]:
        result = [gen_distribution(**params)] * length
    else:
        result = []
        for i in range(length):
            variable = gen_distribution(**params)
            for dependency in dependencies:
                variable += data[dependency][i] * causal_effects["{} -> {}".format(dependency, node)]
            if over_time_effects:
                for dependency in list(over_time_effects):
                    for lag, effect in  enumerate(over_time_effects[dependency]["effects"]):
                        variable += data[dependency][i-1-lag] * effect if (i-1-lag)>=0 else 0 
            result.append(variable)
    return result


def gen_drop_out(data, fraction = None, vacation = None, drop_columns = None):
    """
    This function generates a random drop out.
    :param data: pandas df with patient data
    :param vacation: number of vacation days
    :param fraction: fraction of sampling
    :return: pandas data frame
    """

    if not drop_columns:
        keep_columns = ["patient_id","date","day","Treatment_1","Treatment_2"]
        drop_columns = list(set(list(data.columns))-set(keep_columns))

    treatment_data = data.copy().drop(columns = drop_columns)

    # Apply vacation:
    # Continues period without any data
    if vacation:
        start = np.random.randint(1,len(data)-vacation)
        data = pd.concat([data[:start], data[start+vacation:]])

    # Drop out:
    # random drop out of data 
    if fraction:
        weights_drop_out = 1/(np.array(list(range(len(data))))+1)**2
        weights_drop_out = 1/(np.array(list(range(len(data))))+1)
        data = data.sample(frac = fraction, weights = weights_drop_out)

    # ToDo: Others?
    data = treatment_data.join(data[drop_columns])
    return data.sort_index()


class Simulation:
    def __init__(self, parameter):
        """
        This class handles a simulation
        :param parameter: dictionary with parameters
        """

        self.exposures_params = parameter["exposures"]
        self.outcome_params = parameter["outcome"]

        self.variables = parameter["variables"]
        self.dependencies = extract_dependencies(
            [*self.variables.keys(), *self.exposures_params, self.outcome_params["name"]], parameter["dependencies"])
        self.effect_sizes = parameter["dependencies"]
        if "over_time_dependencies" in parameter.keys():
            self.over_time_dependencies = parameter["over_time_dependencies"]
        else:
            self.over_time_dependencies = {}
        if "drop_out" in parameter.keys():
            self.drop_out = parameter["drop_out"]
        else:
            self.drop_out = {}

    def gen_patient(self, study_design, days_per_period, patient_id = 0, drop_out=None, first_day='2018-01-01'):
        """
        This function generates a person
        :param study_design: study design
        :param days_per_period: days per period
        :return: pandas data frame
        """
        # Define length of study
        length = len(study_design) * days_per_period

        # Generate Dateindex
        dti = pd.date_range(first_day, periods=length, freq='D')

        # Generate Variables with dependencies
        def _order_nodes(dependency_dict):
            final_order = []
            dependencies = [(n, set(dependency_dict[n])) for n in dependency_dict]
            while len(dependencies) > 0:
                dependencies.sort(key=lambda s: len(s[1]))
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
        result = {'patient_id': [patient_id]*length,
                  'date': dti,
                  'day': list(range(1,length+1))}

        for node in ordered_node:
            if node not in list(self.over_time_dependencies.keys()):
                self.over_time_dependencies[node] = None
            # if node is exposure
            if node in list(self.exposures_params.keys()):
                t = simulate_treatment(study_design,
                                       days_per_period,
                                       node,
                                       self.dependencies[node],
                                       self.exposures_params[node],
                                       self.effect_sizes,
                                       result)
                result[node], result["{}_effect".format(node)] = t
            # generate outcome
            elif node == self.outcome_params["name"]:
                o = simulate_outcome(outcome_params=self.outcome_params,
                                     data=result,
                                     length=length,
                                     dependencies=self.dependencies[node],
                                     treatments=list(self.exposures_params.keys()),
                                     causal_effects=self.effect_sizes,
                                     over_time_effects = self.over_time_dependencies[node])
                result["baseline_drift"], result["underlying_state"], result[node] = o
            # if node is variable
            else:
                result[node] = simulate_node(node, self.dependencies[node], length, result, params=self.variables[node],
                                             causal_effects=self.effect_sizes,over_time_effects = self.over_time_dependencies[node])

        data = pd.DataFrame(result)

        if drop_out:
            return data, gen_drop_out(data.copy(), **drop_out)
        return data
