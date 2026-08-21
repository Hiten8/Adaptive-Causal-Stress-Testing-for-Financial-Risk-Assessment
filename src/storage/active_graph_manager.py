"""
Active Graph Manager

Purpose
-------
Maintains the CURRENT ACCEPTED causal state.

Unlike the CausalStateRepository (which stores every PCMCI
window), this manager stores only ONE state:

    The currently accepted causal graph used by the system.

This state is updated only after

    Change Gate
        ↓
Graph Evolution
        ↓
Stress Testing
        ↓
Best Candidate Selected
"""

from pathlib import Path
import joblib
from src.core.causal_state import CausalState

class ActiveGraphManager:

    def __init__(self):
        self.storage_path = Path(
            "results/active_graph/active_state.pkl"
        )

        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    # Exists
    def exists(self):
        return self.storage_path.exists()

    # Save
    def save(self, state: CausalState):
        joblib.dump(
            state,
            self.storage_path
        )

    # Load
    def load(self):
        if not self.exists():
            raise FileNotFoundError(
                "No active causal graph exists.\n"
                "Initialize it using initialize()."
            )

        return joblib.load(self.storage_path)

    # Initialize
    def initialize(self, state: CausalState):
        if self.exists():
            print(
                "Active graph already exists."
            )

            return

        self.save(state)
        print(
            "Active graph initialized."
        )

    # Update
    def update(self, new_state: CausalState):
        self.save(new_state)
        print(
            "Active graph updated."
        )

    # Information
    def info(self):

        if not self.exists():
            print("No active graph.")
            return

        state = self.load()

        print("=" * 60)
        print("Current Active Causal State")
        print("=" * 60)

        print(f"Window ID          : {state.window_id}")
        print(f"Time Period        : {state.start_date} -> {state.end_date}")
        print(f"Dominant Regime    : {state.dominant_regime}")
        print(f"Avg Confidence     : {state.average_confidence:.3f}")
        print(f"Avg Entropy        : {state.average_entropy:.3f}")

        print()
        print(f"Nodes              : {state.graph.number_of_nodes()}")
        print(f"Edges              : {state.graph.number_of_edges()}")

        print()
        print(f"Variables          : {len(state.variable_names)}")
        print("=" * 60)

# Testing
if __name__ == "__main__":

    from src.storage.causal_state_repository import (
        CausalStateRepository
    )

    repository = CausalStateRepository()
    manager = ActiveGraphManager()
    states = repository.load_all()

    if not manager.exists():
        manager.initialize(states[0])

    manager.info()