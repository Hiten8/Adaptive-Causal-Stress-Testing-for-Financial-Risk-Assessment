"""
PCMCI Engine

Runs PCMCI on a single adaptive window and returns
a CausalState object.
"""

import networkx as nx
import numpy as np
import pandas as pd

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from src.core.causal_state import CausalState


# PCMCI PARAMETERS
TAU_MAX = 3
PC_ALPHA = 0.05
ALPHA_LEVEL = 0.05

# PCMCI
def run_pcmci(window_csv, window_metadata):
    """Load a window CSV and run PCMCI on its observations."""
    return run_pcmci_frame(pd.read_csv(window_csv), window_metadata)


def run_pcmci_frame(
    df: pd.DataFrame,
    window_metadata
):
    """
    Runs PCMCI on one adaptive window.

    Parameters
    ----------
    window_csv : str or Path
    window_metadata : dict

    Returns
    -------
    CausalState
    """

    # Work on a copy because bootstrap callers reuse their source frame.
    df = df.copy()

    # Remove metadata columns
    drop_columns = [
        "Date",
        "Regime",
        "Confidence",
        "Entropy"
    ]

    drop_columns.extend(
        [
            c
            for c in df.columns
            if c.startswith("RegimeProb_")
        ]
    )

    X = df.drop(
        columns=drop_columns,
        errors="ignore"
    )

    variable_names = list(X.columns)

    # Tigramite DataFrame
    dataframe = pp.DataFrame(
        X.values,
        var_names=variable_names
    )

    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr()
    )

    results = pcmci.run_pcmci(
        tau_max=TAU_MAX,
        pc_alpha=PC_ALPHA
    )

    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]

    # Build Graph
    graph = nx.DiGraph()
    graph.add_nodes_from(variable_names)
    adjacency_matrix = np.zeros(
        (
            len(variable_names),
            len(variable_names)
        )
    )
    edge_list = []

    # Extract lag-1 causal graph
    for i in range(len(variable_names)):
        for j in range(len(variable_names)):
            if i == j:
                continue

            p = p_matrix[i, j, 1]
            val = val_matrix[i, j, 1]

            if p < ALPHA_LEVEL:
                graph.add_edge(
                    variable_names[i],
                    variable_names[j],
                    weight=float(val),
                    p_value=float(p),
                    lag=1
                )

                adjacency_matrix[i, j] = val
                edge_list.append({
                    "Source": variable_names[i],
                    "Target": variable_names[j],
                    "Lag": 1,
                    "Weight": float(val),
                    "PValue": float(p)
                })

    # Graph Statistics
    graph_metrics = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "average_degree":
            np.mean(
                [
                    d
                    for _, d
                    in graph.degree()
                ]
            )
    }

    # Build CausalState
    causal_state = CausalState(
        # Window
        window_id=window_metadata["WindowID"],
        start_date=window_metadata["StartDate"],
        end_date=window_metadata["EndDate"],

        # Regime
        dominant_regime=window_metadata["DominantRegime"],
        regime_distribution={
            int(k.split("_")[1]): window_metadata[k]
            for k in window_metadata.keys()
            if k.startswith("Regime_")
        },

        average_confidence=
            window_metadata["AverageConfidence"],

        average_entropy=
            window_metadata["AverageEntropy"],

        # Graph
        graph=graph,
        adjacency_matrix=adjacency_matrix,
        edge_strength_matrix=val_matrix,
        p_value_matrix=p_matrix,
        variable_names=variable_names,
        graph_metrics=graph_metrics,
        change_features={},
        candidate_probability=0.0,
        notes={
            "edge_list": edge_list,
            "tau_max": TAU_MAX,
            "pc_alpha": PC_ALPHA,
            "alpha_level": ALPHA_LEVEL
        }
    )

    return causal_state
