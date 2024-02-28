### Created 28.02.2024 by thogaertner (thomas.gaertner@hpi.de)
# Contains functions to simulate dropout

import numpy as np
import pandas as pd

def gen_drop_out(data, fraction = 1, vacation = 0, drop_columns = None, kind="MAR", mnar_weight_column=None):
    """This function generates a random drop out by using pandas.DataFrame.sample(). The weights for MAR were calculated by the time, for MNAR were given through the mnar_weight_column.

    Args:
        data (pandas df): pandas df with patient data
        fraction (float, optional): fraction of sampling. Defaults to None.
        vacation (int, optional): number of vacation days. Defaults to 0.
        drop_columns (list, optional): list of columns, which are deleted for missing values. Defaults to None.
        kind (str, optional): Givin the kind of missingness. Defaults to "MAR".
        mnar_weight_column (str, optional): . Defauls to None

    Raises:
        ValueError: If kind of missing value is not defined.

    Returns:
        pd.DataFrame: Dataframe with missing values.
    """
    # create keep and drop columns
    if not drop_columns:
        keep_columns = ["patient_id","date","day","Treatment_1","Treatment_2"]
        drop_columns = list(set(list(data.columns))-set(keep_columns))

    # assert that drop columns is greater 0
    assert(len(drop_columns)>0)

    treatment_data = data.copy().drop(columns = drop_columns)

    # Apply vacation:
    # Continues period without any data
    if vacation > 0:
        start = np.random.randint(1,len(data)-vacation)
        data = pd.concat([data[:start], data[start+vacation:]])

    # Drop out:
    # random drop out of data 
    if fraction<=1:
        if kind.lower() == "mcar":
            # Function for "missing completly at random"
            data = data.sample(frac = fraction)
        elif kind.lower() == "mar":
            # function for at "missing at random", increase over time
            weights_drop_out = 1/(np.array(list(range(len(data))))+1)
            data = data.sample(frac = fraction, weights = weights_drop_out)
        elif kind.lower() == "mnar":
            # function for "missing not at random"
            weights_drop_out = data[mnar_weight_column]
            data = data.sample(frac = fraction, weights = weights_drop_out)
        else: 
            raise ValueError("Must be MCAR, MAR, or MNAR! Found: {kind.upper()}")

    # Join baseline variables with sampled data
    result_data = treatment_data.join(data[drop_columns])
    # Create col missing
    result_data["missing"] = result_data[drop_columns[0]].isna()
    # Return df
    return result_data.sort_index()
