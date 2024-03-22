import argparse

import json


def create_study_params(file_path: str)->dict:
    """This function creates study parameters based on a daggity text file.

    Args:
        file_path (str): path to daggity.txt

    Returns:
        dict: Study params
    """

    f = open(file_path, "r")
    rows = f.read().split("\n")
    rows = rows[1:-2]

    nodes = []
    outgoing = []
    incoming = []

    def_parmas = {
        "constant": False,
        "distribution": "normal",
        "mean": 0,
        "std": 1,
        "boarders": (-1, 1)}

    exposures = []
    outcomes = []

    for row in rows:
        values = row.split(" ")
        if values[1][0] not in ["-", "<"]:
            if "outcome" in values[1]:
                outcomes.append(values[0])
            elif "exposure" in values[1]:
                exposures.append(values[0])
            nodes.append(values[0])
        else:
            outgoing.append(values[0])
            incoming.append(values[2])

    rows_params = {"exposure_params": {}, "outcome_params": {}, "variable_params": {}, "dependencies": {}, "over_time_dependencies": {}}

    for node in nodes:
        if node in exposures:
            rows_params["exposure_params"][node] = {
                "gamma": 1,
                "tau": 1,
                "treatment_effect": 1}
        elif node in outcomes:
            rows_params["outcome_params"] = {
                "name": node,
                "x_0": {"mu":0, "sigma":1},
                "baseline_drift": {"mu":0, "sigma":1},
                "noise":{"mu": 0,"sigma":1}
                }
        else:
            rows_params["variable_params"][node] = def_parmas
    for out_edge, in_edge in zip(outgoing, incoming):
        if in_edge not in rows_params["dependencies"].keys():
            rows_params["dependencies"][in_edge]={}
        rows_params["dependencies"][in_edge][out_edge] = 1
    return rows_params


def main():
    # Create parser to use this file from command line
    parser = argparse.ArgumentParser(prog='creaty_study_params.py', usage='%(prog)s [input] [output]',
                                     description="Transforms a study parameters file as json out of a dagitty tree.")

    # Input file should be one file containing descr of the daggicty graph.
    parser.add_argument("input", type=str, nargs="*", help="Path to dagitty text file.")
    parser.add_argument("output", type=str, nargs="*", help="Path to output file.")

    args = parser.parse_args()
    res = create_study_params(args.input[0])
    with open(args.input[1], "w") as fp:
        json.dump(res, fp, indent=4)


if __name__ == "__main__":
    main()
