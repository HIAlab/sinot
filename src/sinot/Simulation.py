### Created 28.02.2024 by thogaertner (thomas.gaertner@hpi.de)
# Contains helper functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import extract_dependencies
from .dropout import gen_drop_out
from .Nodes import Node, Outcome, Exposure



# Generate Variables with dependencies
def order_nodes(dependency_dict:dict):
    """Orders the nodes from a dag for simulation. 

    Args:
        dependency_dict (dict): _description_

    Raises:
        ValueError: If the dag could not be resolved, an Error will be raised.

    Returns:
        list: List of sorted nodes based on dag. Nodes should just depend on the previous ones. 
    """
    final_order = []
    dependencies = []
    all_nodes = set(dependency_dict.keys())
    for v in dependency_dict.values():
        all_nodes |= set(v.keys())
    for n in all_nodes:
        dependencies.append((n, set(dependency_dict.get(n,{}).keys())))
        
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


class Patient:
    def __init__(self, outcome_params:dict, exposure_params:dict={}, variable_params:dict={}, dependencies:dict={}, over_time_dependencies:dict={}) -> None:
        """Initialize a patient.

        Args:
            outcome_params (dict): Specifies the outcome with x_0, baseline_drift and noise as keys.
            exposure_params (dict, optional): Specifies the exposures with names as keys and values for gamma, tau and the treatment effect in one dictonary per key. Defaults to {}.
            variable_params (dict, optional): Dictonary with variable names as keys and their parameters as values (dict). Defaults to {}.
            dependencies (dict, optional): Specify the effects from one variable on another. Defaults to {}.
            over_time_dependencies (dict, optional): Specify the effects from one variable on another over time. Defaults to {}.
        """
        self.history = []
        self.outcome_params = outcome_params
        self.exposure_params = exposure_params
        self.variable_params = variable_params
        self.dependency_dict = extract_dependencies([*self.variable_params.keys(), *self.exposure_params.keys(), self.outcome_params["name"]], dependencies)
        self.over_time_dependencies = over_time_dependencies
        self.node_order = order_nodes(dependency_dict=self.dependency_dict)
        self._initialize_nodes()
        pass


    def _initialize_nodes(self):
        """Initalize all nodes with their parameters.
        """
        self.nodes = {}
        # Generate Exposures
        for name, exp_params in zip(self.exposure_params.keys(), self.exposure_params.values()):
            self.nodes[name] = Exposure(name=name, **exp_params)
        # Generate Outcome
        outcome_name = self.outcome_params.get("name","Outcome")
        self.nodes[outcome_name] = Outcome(name=outcome_name, params=self.outcome_params, dependencies=self.dependency_dict.get(outcome_name,{}), overtime_dependencies=self.over_time_dependencies.get(outcome_name, {}))
        for var_name, var_params in zip(self.variable_params.keys(), self.variable_params.values()):
            self.nodes[var_name] = Node(name=var_name, params=var_params, dependencies=self.dependency_dict.get(var_name, {}), overtime_dependencies=self.over_time_dependencies.get(var_name,{}))


    def simulate_next(self, treatment:dict) -> dict:
        """Simulate the next day with a given treatent.

        Args:
            treatment (dict): Specify, which treatment was given. Keys are the treatment names and values define as bool, which treatment was provided. 

        Returns:
            dict: Simulated data of the day.
        """
        current_data = treatment.copy()
        active_exps = [str(t).replace("_flag","") for t in treatment.keys() if treatment[t]]
        current_data["Treatment"] =  active_exps[0] if (len(active_exps)==1) else active_exps
        for n_name in self.node_order:
            current_data.update(self.nodes[n_name].simulate(self.history, current_data))
        self.history.append(current_data)
        return current_data


    def simulate(self, treatment_plan:list) -> pd.DataFrame:
        """Simulate a patient journay with a given treatment plan.

        Args:
            treatment_plan (list): List containing the treatment per day.

        Returns:
            pd.DataFrame: Simulated data
        """
        possible_treatments = set(treatment_plan)
        for t in treatment_plan:
            treatment = {f"{_t}_flag": False for _t in possible_treatments}          
            treatment[f"{t}_flag"] = True
            self.simulate_next(treatment=treatment)
        return pd.DataFrame(self.history)
    

    def plot(self, title="Plot"):
        """Plot the data of the patients

        Args:
            title (str, optional): Title of the plot. Defaults to "Plot".
        """
        patient_data = pd.DataFrame(self.history)
        outcome_name = self.outcome_params["name"]
        fig, axs = plt.subplots(ncols=2, figsize=(10,4))
        axs[0].plot(patient_data[outcome_name], label="Observed Outcome", linestyle="", marker="x")
        axs[0].plot(patient_data[f"{outcome_name}_underlying_state"], label="Underlying State")
        axs[0].plot(patient_data[f"{outcome_name}_baseline_drift"], label="Baseline Drift")
        axs[0].legend()
        axs[0].set_title("Outcome")

        for exposure_name in self.exposure_params.keys():
            axs[1].plot(patient_data[exposure_name], label=exposure_name)
        axs[1].plot(patient_data[self.exposure_params.keys()].sum(axis=1), linestyle=":", label="Overall Treatment Effect")
        axs[1].legend()
        axs[1].set_title("Treatment")
        fig.suptitle(title)
        plt.show()


class Trial:
    def __init__(self, outcome_params:dict, exposure_params:dict={}, variable_params:dict={}, dependencies:dict={}, over_time_dependencies:dict={}):
        """Initialize a study.

        Args:
            outcome_params (dict): Specifies the outcome with x_0, baseline_drift and noise as keys.
            exposure_params (dict, optional): Specifies the exposures with names as keys and values for gamma, tau and the treatment effect in one dictonary per key. Defaults to {}.
            variable_params (dict, optional): Dictonary with variable names as keys and their parameters as values (dict). Defaults to {}.
            dependencies (dict, optional): Specify the effects from one variable on another. Defaults to {}.
            over_time_dependencies (dict, optional): Specify the effects from one variable on another over time. Defaults to {}.
        """
        self.exposure_params = exposure_params
        self.outcome_params = outcome_params

        self.variable_params = variable_params
        self.dependencies = extract_dependencies(
            [*self.variable_params.keys(), *self.exposure_params.keys(), self.outcome_params["name"]], dependencies)
        self.effect_sizes = dependencies
        self.over_time_dependencies = over_time_dependencies


    def simulate_patient(self, treatment_plan:list, patient_id = 0, dropout_params:dict=None, first_day:str='2018-01-01') -> tuple:
        """Simulate one patient of the study

        Args:
            treatment_plan (list): Specify the given treatment per day.
            patient_id (int, optional): Unique identifier of the patient. Defaults to 0.
            dropout_params (dict, optional): Paramters for dropout. Defaults to None.
            first_day (str, optional): Day of study start. Defaults to '2018-01-01'.

        Returns:
            tuple: two Pandas Dataframes containing complete patient data and data with dropout.
        """
        nb_observations = len(treatment_plan)
                
        # Generate Dateindex, Treatmentvariable and bloc
        dti = pd.date_range(first_day, periods=nb_observations, freq='D')
        

        # Generate Data
        result = {'patient_id': [patient_id]*nb_observations,
                    'date': dti,
                    'day_of_study': list(range(1,nb_observations+1))}

        data = pd.DataFrame(result)
        data["Block"] = (np.array(treatment_plan) != np.array([""]+treatment_plan[:-1])).astype(int).cumsum()
        start_day_of_block = data.groupby("Block").agg({"day_of_study":"min"})
        data["day_of_block"] = data.apply(lambda x: 1+x["day_of_study"] - start_day_of_block.loc[x["Block"],"day_of_study"], axis=1)


        patient = Patient(
            exposure_params=self.exposure_params, 
            outcome_params=self.outcome_params, 
            variable_params=self.variable_params, 
            dependencies=self.dependencies, 
            over_time_dependencies=self.over_time_dependencies)

        data = pd.concat([data, pd.DataFrame(patient.simulate(treatment_plan))], axis=1)

        if dropout_params:
            drop_out_data = gen_drop_out(data.copy(), **dropout_params)
        else: 
            drop_out_data = data.copy()
        return data, drop_out_data


    def simulate_nof1_study(self, treatment_plan:list, nb_patients:int, start_id=1, dropout_params:dict=None, first_day='2018-01-01')->tuple:
        """Creates a nof1 study with multiple patients

        Args:
            treatment_plan (list): Specify the given treatment per day.
            nb_patients (int): number of patients.
            start_id (int, optional): First id of the patient. Will be increased by 1 per patient. Defaults to 1.
            dropout_params (dict, optional): Paramters for dropout. Defaults to None.
            first_day (str, optional): Day of study start. Defaults to '2018-01-01'.

        Returns:
            tuple: two Pandas Dataframes containing complete patient data and data with dropout.
        """
        patient_data_complete = []
        patient_data_dropout = []
        for patient_id in range(start_id, nb_patients+start_id):
            _dat_complete, _dat_dropout = self.simulate_patient(treatment_plan=treatment_plan, patient_id=patient_id, drop_out=dropout_params, first_day=first_day)
            patient_data_complete.append(_dat_complete)
            patient_data_dropout.append(_dat_dropout)
        return pd.concat(patient_data_complete, axis=0).reset_index(drop=True), pd.concat(patient_data_dropout, axis=0).reset_index(drop=True)
        

    def plot_data(self, patient_df:pd.DataFrame, title:str="Plot"):
        """Plot the data of the first patient from a given study.

        Args:
            patient_df (pd.DataFrame): study data. Can be complete or with missing values
            title (str, optional): Title of the plot. Defaults to "Plot".
        """
        patient_df = patient_df[patient_df["patient_id"]==patient_df["patient_id"].unique()[0]]
        outcome_name = self.outcome_params["name"]
        fig, axs = plt.subplots(ncols=2, figsize=(10,4))
        axs[0].plot(patient_df[outcome_name], label="Observed Outcome", linestyle="", marker="x")
        axs[0].plot(patient_df[f"{outcome_name}_underlying_state"], label="Underlying State")
        axs[0].plot(patient_df[f"{outcome_name}_baseline_drift"], label="Baseline Drift")
        axs[0].legend()
        axs[0].set_title("Outcome")

        for exposure_name in self.exposure_params.keys():
            axs[1].plot(patient_df[exposure_name], label=exposure_name)
        axs[1].plot(patient_df[self.exposure_params.keys()].sum(axis=1), linestyle=":", label="Overall Treatment Effect")
        axs[1].legend()
        axs[1].set_title("Treatment")
        fig.suptitle(title)
        plt.show()