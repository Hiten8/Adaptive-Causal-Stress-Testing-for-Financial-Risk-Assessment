"""
Repository for storing and retrieving CausalState objects.
"""

from pathlib import Path
from typing import List
import joblib
from src.core.causal_state import CausalState

class CausalStateRepository:
    def __init__(self,
                 storage_folder="results/causal_states"):
        self.storage_folder = Path(storage_folder)
        self.storage_folder.mkdir(
            parents=True,
            exist_ok=True
        )

    # Private
    def _filename(self,
                  window_id: int):
        return self.storage_folder / (
            f"causal_state_{window_id:04d}.pkl"
        )
    
    # Save
    def save(self,
             causal_state: CausalState):
        
        filename = self._filename(
            causal_state.window_id
        )
        joblib.dump(
            causal_state,
            filename
        )

    # Load
    def load(self,
             window_id: int):

        filename = self._filename(window_id)
        if not filename.exists():
            raise FileNotFoundError(
                f"No CausalState found for "
                f"window {window_id}"
            )
        return joblib.load(filename)

    # Exists
    def exists(self,
               window_id: int):

        return self._filename(
            window_id
        ).exists()

    # Delete
    def delete(self,
               window_id: int):

        filename = self._filename(
            window_id
        )

        if filename.exists():
            filename.unlink()

    # List IDs
    def list_window_ids(self) -> List[int]:
        ids = []
        for file in self.storage_folder.glob(
                "causal_state_*.pkl"):
            window_id = int(
                file.stem.split("_")[-1]
            )

            ids.append(window_id)

        return sorted(ids)

    # Load All
    def load_all(self):
        states = []
        for window_id in self.list_window_ids():
            states.append(
                self.load(window_id)
            )

        return states

    # Count
    def count(self):
        return len(
            self.list_window_ids()
        )

    # Clear Repository
    def clear(self):
        for file in self.storage_folder.glob(
                "*.pkl"):
            file.unlink()

    # Latest State
    def latest(self):
        ids = self.list_window_ids()
        if len(ids) == 0:
            return None

        return self.load(ids[-1])

    # Previous State
    def previous(self,
                 window_id: int):

        ids = self.list_window_ids()
        if window_id not in ids:
            return None
        index = ids.index(window_id)
        if index == 0:
            return None

        return self.load(
            ids[index - 1]
        )