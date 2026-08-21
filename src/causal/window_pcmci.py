"""
Run PCMCI on all Adaptive Regime-Aware Windows.

For each window:
    1. Run PCMCI
    2. Create CausalState
    3. Compute Graph Fingerprint
    4. Save using Repository
    5. Build summary report
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.causal.pcmci_engine import run_pcmci
from src.core.graph_fingerprint import compute_graph_fingerprint
from src.storage.causal_state_repository import CausalStateRepository

WINDOW_FOLDER = Path("data/windows")

METADATA_FILE = Path("data/metadata/window_metadata.csv")

SUMMARY_FILE = Path(
    "results/causal_states/causal_state_summary.csv"
)

WINDOW_SIZE = 120
STEP_SIZE = 1

# Repository
repository = CausalStateRepository()

# Load Metadata
metadata = pd.read_csv(METADATA_FILE)
summary = []

print("=" * 70)
print("Running PCMCI on Adaptive Windows")
print("=" * 70)


# Loop Through Every Window
for _, row in tqdm(
    metadata.iterrows(),
    total=len(metadata),
    desc="Processing Windows"
):

    window_file = WINDOW_FOLDER / row["File"]
    window_metadata = row.to_dict()

    # PCMCI
    causal_state = run_pcmci(
        window_csv=window_file,
        window_metadata=window_metadata
    )

    # # Store window parameters
    # causal_state.window_size = WINDOW_SIZE
    # causal_state.step_size = STEP_SIZE

    # Graph Fingerprint
    causal_state.graph_fingerprint = compute_graph_fingerprint(
        graph=causal_state.graph,
        adjacency_matrix=causal_state.adjacency_matrix,
        edge_strength_matrix=causal_state.edge_strength_matrix,
        regime=causal_state.dominant_regime,
        confidence=causal_state.average_confidence,
        entropy=causal_state.average_entropy
    )

    repository.save(causal_state)

    # Summary
    fp = causal_state.graph_fingerprint
    summary.append({
        "WindowID":
            causal_state.window_id,
        "StartDate":
            causal_state.start_date,
        "EndDate":
            causal_state.end_date,
        "DominantRegime":
            causal_state.dominant_regime,
        "AverageConfidence":
            causal_state.average_confidence,
        "AverageEntropy":
            causal_state.average_entropy,

        "Nodes":
            fp["num_nodes"],

        "Edges":
            fp["num_edges"],

        "Density":
            fp["density"],

        "AverageInDegree":
            fp["avg_in_degree"],

        "AverageOutDegree":
            fp["avg_out_degree"],

        "MaximumInDegree":
            fp["max_in_degree"],

        "MaximumOutDegree":
            fp["max_out_degree"],

        "AverageEdgeWeight":
            fp["avg_edge_weight"],

        "MaximumEdgeWeight":
            fp["max_edge_weight"],

        "EdgeWeightStd":
            fp["edge_weight_std"],

        "ConnectedComponents":
            fp["connected_components"],

        "Clustering":
            fp["clustering"],

        "AdjacencySparsity":
            fp["adjacency_sparsity"],

        "MeanEdgeStrength":
            fp["mean_edge_strength"]

    })

summary_df = pd.DataFrame(summary)
SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

print()
print("=" * 70)
print("PCMCI Processing Complete")
print("=" * 70)
print()
print(f"Total Windows Processed : {len(summary_df)}")
print(f"Causal States Saved     : {repository.count()}")
print()
print(summary_df.head())
print()
print(f"Summary saved to:\n{SUMMARY_FILE}")