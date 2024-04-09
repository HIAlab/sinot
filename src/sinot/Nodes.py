### Created 28.02.2024 by thogaertner (thomas.gaertner@hpi.de)
# Contains functions for simulating a node

from .generate_distributions import gen_distribution
import numpy as np


DEFAULT_OUTCOME_PARAMS = dict(
    x_0 = {"mu":0, "sigma":0},     # Specifies the start point
    baseline_drift={"mu": 0,"sigma":0},   # Specifies the baseline drift
    noise={"mu": 0,"sigma":0} # Specifies Noise
)


class AbstractVariable:
    name = "C"
    params = {}
    
    def __init__(self, name:str, params:dict={}) -> None:
        """Initialize an object.

        Args:
            name (str): Name of the node.
            params (dict, optional): Parameter of the node. Defaults to {}.
        """
        self.name = name
        self.params = params
        pass

    def _simulate_dependencies(self, variable:float, current_data:dict) -> float:
        """Simulate (linear) dependencies within the node depending on the value of variable and current data.

        Args:
            variable (float): value of the variable
            current_data (dict): data at the current timepoint.

        Returns:
            float: variable with effects.
        """
        for dependency in self.dependencies.keys():
            variable += current_data[dependency] * self.dependencies[dependency]
        return variable


    def _simulate_over_time_dependencies(self, variable:float, history:list):
        """Simulate (linear) over-time-dependencies within the node depending on the value of variable and current data.

        Args:
            variable (float): value of the variable
            history (list): data until the current timepoint.

        Returns:
            float: variable with effects.
        """
        for dependency in self.overtime_dependencies.keys():
            # Iterate over all lags
            for lag, effect in  enumerate(self.overtime_dependencies[dependency]):
                # Check, if data is avialable
                flag_lagged_value_in_study_period = len(history) > lag
                # Get the data of the dependeny
                if (flag_lagged_value_in_study_period):
                    dependent_data = float(history[-(lag+1)][dependency])
                    variable +=  float(dependent_data * effect)
        return variable


    def _check_variable(self, variable:float, only_boarders:bool=False) -> float:
        """Check, if the variable is valid in the boarders and digits.

        Args:
            variable (float): Value of the variable
            only_boarders (bool, optional): Defines, if only the boardes should be checked. Defaults to False.

        Returns:
            float: value of the varible within the boarders
        """
        if ("digits" in self.params.keys()) and not(only_boarders):
            variable = np.round(variable, self.params["digits"])
        if "boarders" in self.params.keys():
            if self.params["boarders"][0]:
                variable = self.params["boarders"][0] if variable<self.params["boarders"][0] else variable 
            if self.params["boarders"][1]:
                variable = self.params["boarders"][1] if variable>self.params["boarders"][1] else variable 
        return variable



class Node(AbstractVariable):
    def __init__(self, name: str, params: dict, dependencies={}, overtime_dependencies={}) -> None:
        """Initialization of a node object, which are used for covariates.

        Args:
            name (str): name of the node
            params (dict): contains parameters for simulating the nodes such as distribution params.
            dependencies (dict, optional): Specify effect of variables on this node. Defaults to {}.
            overtime_dependencies (dict, optional): Specify over time effects of variables on this node. Defaults to {}.
        """
        super().__init__(name, params)
        self.dependencies = dependencies
        self.overtime_dependencies = overtime_dependencies
        if self.params.get("constant"):
            self.constant_variable = gen_distribution(**self.params)


    def simulate(self, history:list, current_data:dict) -> dict:
        """Simulate node based on current data.

        Args:
            history (list): history of the study
            current_data (dict): contains the values of all variables at the current time point.

        Returns:
            dict: value for the variable.
        """
        if self.params.get("constant"):
            return {self.name: self.constant_variable}
        else:
            variable = gen_distribution(**self.params)
            variable = self._simulate_dependencies(variable, current_data)
            variable = self._simulate_over_time_dependencies(variable, history)
            variable = self._check_variable(variable)
            return {self.name: variable}
        

class Exposure(AbstractVariable):
    def __init__(self, name: str, gamma:int=1, tau:int=1, treatment_effect:float=1) -> None:
        """Initialize an Exposure Variable

        Args:
            name (str): Name of the exposure variable.
            gamma (int, optional): Wash-in parameter. Defaults to 1.
            tau (int, optional): Wash-out parameter. Defaults to 1.
            treatment_effect (float, optional): effect size. Defaults to 1.
        """
        super().__init__(name, {})
        self.gamma = gamma
        self.tau = tau
        self.treatment_effect = treatment_effect


    def simulate(self, history:list, current_data:dict) -> dict:
        """Simulate exposure based on current data.

        Args:
            history (list): history of the study
            current_data (dict): contains the values of all variables at the current time point.

        Returns:
            dict: value for the variable.
        """
        treatment = current_data.get(f"{self.name}_flag", False)
        if (len(history)==0):
            x_j=0
        else:
            x_j = history[-1].get(self.name, 0)
        t = 1 if treatment else 0
        x_j +=  ((self.treatment_effect - x_j) / self.tau) * t - (x_j / self.gamma) * (1 - t)
        return {self.name: x_j}


class Outcome(Node):
    def __init__(self, name: str, params: dict, dependencies={}, overtime_dependencies={}) -> None:
        """Initialize an Exposure Variable

        Args:
            name (str): Name of the outcome variable.
            params (dict): contains parameters for simulating the nodes such x_0, noise, and baseline_drift params.
            dependencies (dict, optional): Specify effect of variables on this node. Defaults to {}.
            overtime_dependencies (dict, optional): Specify over time effects of variables on this node. Defaults to {}.
        """
        # Copy the default params and update the dict, so that changes from the user will be applied, but all parameters were set. 
        default_params = DEFAULT_OUTCOME_PARAMS.copy()
        default_params.update(params)
        # Super init
        super().__init__(name, default_params, dependencies, overtime_dependencies)


    def _simulate_baseline_drift(self, history:list, mu:float, sigma:float) -> float:
        """Simulates a baselinedrift following the wiener prozess (AR1).

        Args:
            history (list): variables simulated until now.
            mu (float): Mean of the normal distributed baseline drift.
            sigma (float): Std of the normal distributed baseline drift.

        Returns:
            float: simulated value.
        """
        if (len(history) == 0):
            x_0 = gen_distribution(**self.params.get("x_0"))
        else:
            x_0 = history[-1][f"{self.name}_baseline_drift"]
            x_0 += np.random.normal(mu, sigma)
            x_0 = self._check_variable(x_0, only_boarders=True)
        return x_0
    

    def simulate(self, history:list, current_data:dict) -> dict:
        """Simulate Outcome based on current data.

        Args:
            history (list): history of the study
            current_data (dict): contains the values of all variables at the current time point.

        Returns:
            dict: value for the variable.
        """
        # Generate baseline drift
        bd_params = self.params.get("baseline_drift", {})
        x_bd = self._simulate_baseline_drift(history=history, mu=bd_params.get("mu",0), sigma=bd_params.get("sigma",0))
        # Generate underlying state
        x_us = self._simulate_dependencies(x_bd, current_data=current_data)
        x_us = self._simulate_over_time_dependencies(x_us, history=history)
        # generate observation
        noise_params = self.params.get("noise",{"distribution":"normal", "mu":0, "sigma":0})
        x_o = x_us + gen_distribution(**noise_params)
        x_o = self._check_variable(x_o)
        return {f"{self.name}_baseline_drift": x_bd, f"{self.name}_underlying_state": x_us, self.name: x_o}