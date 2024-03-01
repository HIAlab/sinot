### Created 28.02.2024 by thogaertner (thomas.gaertner@hpi.de)
# Contains helper functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import extract_dependencies
from .dropout import gen_drop_out
from .sim_node import simulate_node
from .sim_outcome import simulate_outcome
from .sim_treatment import simulate_treatment


class Simulation:
    def __init__(self, parameter):
        """
        This class handles a simulation
        :param parameter: dictionary with parameters
        """

        self.exposures_params = parameter["exposures"]
        self.outcome_params = parameter["outcome"]

        self.variables = parameter.get("variables", {})
        self.dependencies = extract_dependencies(
            [*self.variables.keys(), *self.exposures_params.keys(), self.outcome_params["name"]], parameter["dependencies"])
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

        # Generate Dateindex, Treatmentvariable and bloc
        dti = pd.date_range(first_day, periods=length, freq='D')
        treatment = sum([[t]*days_per_period for t in study_design], [])
        block = sum([[i+1]*days_per_period for i, _ in enumerate(study_design)], [])


        # Generate Variables with dependencies
        def _order_nodes(dependency_dict):
            final_order = []
            dependencies = [(n, set(dependency_dict[n])) for n in dependency_dict]
            while len(dependencies) > 0:
                dependencies.sort(key=lambda s: len(s[1]))
                # if no dependency is present, we can add it to the final model
                if len(dependencies[0][1]) == 0:
                    n = dependencies.pop(0)[0]
                    final_order.append(n)
                    for dep in dependencies:
                        dep[1].discard(n)
                else:
                    raise ValueError(f"Not possible to order nodes. Could not resolve {dependencies[0][1]}. Please double check, if dependencies following a direct-acyclic-graph (DAG).")
            return final_order

        # Order variables based on their dependencies
        ordered_node = _order_nodes(self.dependencies)

        # Generate Data
        result = {'patient_id': [patient_id]*length,
                  'date': dti,
                  'block': block,
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
                                             causal_effects=self.effect_sizes,
                                             over_time_effects=self.over_time_dependencies[node])

        data = pd.DataFrame(result)
        data['Treatment'] = treatment

        if drop_out:
            return data, gen_drop_out(data.copy(), **drop_out)
        return data


    def plot_patient(self, patient):
        """
        This functions plot a random patient.
        :return: None
        """
        plt.figure(dpi=(100))
        plt.plot(patient["baseline_drift"], label="Baseline")
        plt.plot(patient["underlying_state"], label="Underlying State")
        plt.plot(patient[self.outcome_params["name"]], label="Observation", linestyle="", marker="x")
        plt.legend(loc='lower left')
        plt.xlabel("Day")
        plt.ylabel("Outcome")
        plt.show()
        plt.figure(dpi=(100))

        overall_treatment_effect = [0]*len(patient)

        for t in self.exposures_params:
            plt.plot(patient["{}_effect".format(t)], label=t)
            overall_treatment_effect += patient["{}_effect".format(t)]
        plt.plot(np.array(overall_treatment_effect),
                 label="Overall Treatment Effect", linestyle=":")
        plt.legend(loc='lower left')
        plt.xlabel("Day")
        plt.ylabel("Treatment Effect")
        plt.show()