"""
Graph Update Module
Final graph-evolution commitment stage.

Pipeline

    Change Gate
         |
         v
    Interaction Engine
         |
         v
    Graph Evolution Engine
         |
         v
    Candidate Graphs
         |
         v
    Candidate Ranking
         |
         v
    Rank-1 Candidate
         |
         v
    Graph Update Engine
         |
         v
    Updated Base Graph


Important

The Rank-1 candidate is NOT used as a replacement for the
stored/base graph.

Instead:
    Updated Base Graph
        =
    Original Base Graph
        +
    Rank-1 Evolution
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from src.core.causal_state import CausalState

from src.storage.active_graph_manager import (
    ActiveGraphManager
)

from src.storage.causal_state_repository import (
    CausalStateRepository
)

from src.change_gate.change_gate import (
    ChangeGate
)

from src.graph_evolution.graph_evolution_engine import (
    GraphEvolutionEngine
)

from src.graph_evolution.candidate_ranking import (
    CandidateRankingEngine
)


class GraphUpdateEngine:
    """
    Applies the Rank-1 graph evolution incrementally to the
    original/base causal graph.

    This class does not generate candidate graphs and does not
    perform candidate ranking.

    Those responsibilities belong to:
        GraphEvolutionEngine
        CandidateRankingEngine
    """

    def __init__(
        self,
        weight_tolerance: float = 1e-12
    ):
        """
        Parameters
        ----------
        weight_tolerance : float
            Numerical tolerance for comparing edge weights.
        """

        self.weight_tolerance = (
            weight_tolerance
        )

    # Public Update API
    def update(
        self,
        base_state: CausalState,
        selected_candidate: CausalState
    ) -> CausalState:
        """
        Apply the selected Rank-1 evolution to the base state.

        Parameters
        base_state : CausalState
            Original accepted/base causal state.

        selected_candidate : CausalState
            Rank-1 candidate selected by the ranking engine.

        Returns
        CausalState
            New updated causal state.

        Notes
        base_state is NOT modified in place.
        """

        if base_state is None:
            raise ValueError(
                "base_state cannot be None."
            )

        if selected_candidate is None:
            raise ValueError(
                "selected_candidate cannot be None."
            )

        if not hasattr(
            base_state,
            "graph"
        ):
            raise AttributeError(
                "base_state does not contain a graph."
            )

        if not hasattr(
            selected_candidate,
            "graph"
        ):
            raise AttributeError(
                "selected_candidate does not contain a graph."
            )

        updated_state = deepcopy(
            base_state
        )

        base_graph = (
            base_state.graph
        )

        candidate_graph = (
            selected_candidate.graph
        )

        updated_graph = deepcopy(
            base_graph
        )

        changes = (
            self._detect_graph_differences(
                base_graph,
                candidate_graph
            )
        )

        self._apply_edge_additions(
            updated_graph,
            changes[
                "added_edges"
            ],
            candidate_graph
        )

        self._apply_edge_removals(
            updated_graph,
            changes[
                "removed_edges"
            ]
        )

        self._apply_weight_changes(
            updated_graph,
            changes[
                "weight_changes"
            ]
        )

        updated_state.graph = (
            updated_graph
        )

        self._record_update_metadata(
            updated_state,
            changes,
            selected_candidate
        )

        return updated_state


    def _detect_graph_differences(
        self,
        base_graph: nx.DiGraph,
        candidate_graph: nx.DiGraph
    ) -> Dict:
        """
        Determine the exact graph evolution represented by the
        selected Rank-1 candidate.

        Returns
        -------
        dict

            added_edges
            removed_edges
            weight_changes
        """

        base_edges = set(
            base_graph.edges()
        )

        candidate_edges = set(
            candidate_graph.edges()
        )

        added_edges = sorted(
            candidate_edges - base_edges,
            key=str
        )

        removed_edges = sorted(
            base_edges - candidate_edges,
            key=str
        )

        weight_changes = []

        common_edges = (
            base_edges
            &
            candidate_edges
        )

        for u, v in sorted(
            common_edges,
            key=str
        ):
            old_weight = (
                self._get_edge_weight(
                    base_graph,
                    u,
                    v
                )
            )

            new_weight = (
                self._get_edge_weight(
                    candidate_graph,
                    u,
                    v
                )
            )

            if not self._weights_equal(
                old_weight,
                new_weight
            ):
                weight_changes.append(
                    {
                        "edge":
                            (u, v),

                        "old_weight":
                            old_weight,

                        "new_weight":
                            new_weight
                    }
                )

        return {

            "added_edges":
                added_edges,

            "removed_edges":
                removed_edges,

            "weight_changes":
                weight_changes
        }
    

    def _apply_edge_additions(
        self,
        graph: nx.DiGraph,
        added_edges: List[Tuple],
        candidate_graph: nx.DiGraph
    ) -> None:
        """
        Apply edge additions from the selected candidate.
        """

        for u, v in added_edges:
            if graph.has_edge(
                u,
                v
            ):

                continue

            attributes = deepcopy(
                candidate_graph[
                    u
                ][
                    v
                ]
            )

            graph.add_edge(
                u,
                v,
                **attributes
            )


    def _apply_edge_removals(
        self,
        graph: nx.DiGraph,
        removed_edges: List[Tuple]
    ) -> None:
        """
        Apply edge removals from the selected candidate.
        """

        for u, v in removed_edges:
            if graph.has_edge(
                u,
                v
            ):
                graph.remove_edge(
                    u,
                    v
                )


    def _apply_weight_changes(
        self,
        graph: nx.DiGraph,
        weight_changes: List[Dict]
    ) -> None:
        """
        Apply edge-weight modifications.
        """

        for change in weight_changes:
            u, v = (
                change[
                    "edge"
                ]
            )

            new_weight = (
                change[
                    "new_weight"
                ]
            )

            if not graph.has_edge(
                u,
                v
            ):
                continue

            graph[u][v][
                "weight"
            ] = new_weight


    def _get_edge_weight(
        self,
        graph: nx.DiGraph,
        u,
        v
    ) -> Any:
        """
        Get an edge's weight.
        """

        if not graph.has_edge(
            u,
            v
        ):

            return None

        return graph[u][v].get(
            "weight"
        )

    def _weights_equal(
        self,
        old_weight: Any,
        new_weight: Any
    ) -> bool:
        """
        Compare two weights safely.
        """

        if (
            old_weight is None
            and
            new_weight is None
        ):

            return True

        if (
            old_weight is None
            or
            new_weight is None
        ):

            return False

        try:
            return (
                abs(
                    float(
                        old_weight
                    )
                    -
                    float(
                        new_weight
                    )
                )
                <=
                self.weight_tolerance
            )

        except (
            TypeError,
            ValueError
        ):
            return (
                old_weight
                ==
                new_weight
            )


    def _record_update_metadata(
        self,
        updated_state: CausalState,
        changes: Dict,
        selected_candidate: CausalState
    ) -> None:
        """
        Record information about the selected evolution.

        This preserves traceability from the updated state back
        to the interaction plan and propagation rule that
        generated the Rank-1 candidate.
        """

        if not hasattr(
            updated_state,
            "notes"
        ):

            updated_state.notes = {}

        candidate_notes = getattr(
            selected_candidate,
            "notes",
            {}
        )

        if not isinstance(
            candidate_notes,
            dict
        ):

            candidate_notes = {}

        updated_state.notes[
            "GraphUpdate"
        ] = {

            "AddedEdges":
                deepcopy(
                    changes[
                        "added_edges"
                    ]
                ),

            "RemovedEdges":
                deepcopy(
                    changes[
                        "removed_edges"
                    ]
                ),

            "WeightChanges":
                deepcopy(
                    changes[
                        "weight_changes"
                    ]
                ),

            "SourceInteractionPlan":
                candidate_notes.get(
                    "InteractionPlan",
                    "N/A"
                ),

            "SourceInteractionLevel":
                candidate_notes.get(
                    "InteractionLevel",
                    "N/A"
                ),

            "SourcePropagationRule":
                candidate_notes.get(
                    "PropagationRule",
                    "N/A"
                ),

            "SourceAction":
                candidate_notes.get(
                    "Action",
                    "N/A"
                )
        }

    def summarize_update(
        self,
        base_state: CausalState,
        updated_state: CausalState
    ) -> Dict:
        """
        Summarize the differences between the original base
        state and the updated state.
        """

        changes = (
            self._detect_graph_differences(
                base_state.graph,
                updated_state.graph
            )
        )

        added_count = len(
            changes[
                "added_edges"
            ]
        )

        removed_count = len(
            changes[
                "removed_edges"
            ]
        )

        weight_count = len(
            changes[
                "weight_changes"
            ]
        )

        return {
            "AddedEdges":
                changes[
                    "added_edges"
                ],

            "RemovedEdges":
                changes[
                    "removed_edges"
                ],

            "WeightChanges":
                changes[
                    "weight_changes"
                ],

            "AddedEdgeCount":
                added_count,

            "RemovedEdgeCount":
                removed_count,

            "WeightChangeCount":
                weight_count,

            "TotalEvolutionOperations":
                (
                    added_count
                    +
                    removed_count
                    +
                    weight_count
                )
        }

    def print_update_summary(
        self,
        base_state: CausalState,
        updated_state: CausalState,
        rank: Optional[int] = None,
        candidate_index: Optional[int] = None,
        final_score: Optional[float] = None,
        interaction_plan: Optional[str] = None,
        propagation_rule: Optional[str] = None
    ) -> None:
        """
        Print the final selected evolution and resulting graph
        update.
        """

        summary = (
            self.summarize_update(
                base_state,
                updated_state
            )
        )

        print()

        print(
            "=" * 80
        )

        print(
            "FINAL GRAPH EVOLUTION"
        )

        print(
            "=" * 80
        )

        print()

        if rank is not None:
            print(
                f"Selected Rank : "
                f"{rank}"
            )

        if candidate_index is not None:
            print(
                f"Original Candidate ID : "
                f"{candidate_index}"
            )

        if final_score is not None:
            print(
                f"Final Score : "
                f"{final_score:.6f}"
            )

        if interaction_plan is not None:
            print()

            print(
                f"Interaction Plan : "
                f"{interaction_plan}"
            )

        if propagation_rule is not None:
            print(
                f"Propagation Rule : "
                f"{propagation_rule}"
            )

        print()

        print(
            "-" * 80
        )

        print(
            "APPLIED EVOLUTION"
        )

        print(
            "-" * 80
        )

        print()

        print(
            f"Added Edges : "
            f"{summary['AddedEdgeCount']}"
        )

        for edge in (
            summary[
                "AddedEdges"
            ]
        ):

            print(
                f"   + {edge[0]} "
                f"-> {edge[1]}"
            )

        print()

        print(
            f"Removed Edges : "
            f"{summary['RemovedEdgeCount']}"
        )

        for edge in (
            summary[
                "RemovedEdges"
            ]
        ):
            print(
                f"   - {edge[0]} "
                f"-> {edge[1]}"
            )

        print()

        print(
            f"Weight Changes : "
            f"{summary['WeightChangeCount']}"
        )

        for change in (
            summary[
                "WeightChanges"
            ]
        ):
            u, v = (
                change[
                    "edge"
                ]
            )

            print(
                f"   ~ {u} -> {v}"
            )

            print(
                f"      Old Weight : "
                f"{change['old_weight']}"
            )

            print(
                f"      New Weight : "
                f"{change['new_weight']}"
            )

        print()

        print(
            f"Total Evolution Operations : "
            f"{summary['TotalEvolutionOperations']}"
        )

        print()

        print(
            "-" * 80
        )

        print(
            "UPDATED BASE GRAPH"
        )

        print(
            "-" * 80
        )

        print()

        print(
            f"Original Edge Count : "
            f"{base_state.graph.number_of_edges()}"
        )

        print(
            f"Updated Edge Count  : "
            f"{updated_state.graph.number_of_edges()}"
        )

        print()

        for edge in sorted(
            updated_state.graph.edges(),
            key=str
        ):
            u, v = edge

            weight = (
                updated_state.graph[
                    u
                ][
                    v
                ].get(
                    "weight"
                )
            )

            if weight is None:
                print(
                    f"   {u} -> {v}"
                )

            else:
                print(
                    f"   {u} -> {v} "
                    f"[weight={weight}]"
                )

        print()

        print(
            "=" * 80
        )


def run_pipeline(
    commit: bool = False
):
    """
    Execute the complete graph-evolution pipeline up to the
    graph-update stage.

    Persistence is intentionally NOT performed yet.
    """

    print()

    print(
        "=" * 80
    )

    print(
        "ADAPTIVE CAUSAL GRAPH UPDATE PIPELINE"
    )

    print(
        "=" * 80
    )

    print()

    # STEP 1
    # LOAD BASE / CURRENT STATE
    manager = (
        ActiveGraphManager()
    )

    repository = (
        CausalStateRepository()
    )

    base_state = (
        manager.load()
    )

    if base_state is None:
        raise RuntimeError(
            "No stored/base causal state was found."
        )

    current_states = (
        repository.load_all()
    )

    if not current_states:
        raise RuntimeError(
            "No current causal states were found."
        )

    # Keep the same current-state convention used by the
    # existing testing pipeline.

    current_state = (
        current_states[1]
        if len(
            current_states
        ) > 1
        else current_states[0]
    )

    print(
        "Base state loaded."
    )

    print(
        f"Base Graph Edges : "
        f"{base_state.graph.number_of_edges()}"
    )

    print()

    # STEP 2
    # CHANGE GATE
    print(
        "-" * 80
    )

    print(
        "STEP 1 : CHANGE GATE"
    )

    print(
        "-" * 80
    )

    gate = (
        ChangeGate()
    )

    decision = (
        gate.evaluate(
            base_state,
            current_state
        )
    )

    # STEP 3
    # GRAPH EVOLUTION ENGINE
    print()

    print(
        "-" * 80
    )

    print(
        "STEP 2 : GRAPH EVOLUTION"
    )

    print(
        "-" * 80
    )

    evolution_engine = (
        GraphEvolutionEngine()
    )

    candidates = (
        evolution_engine.generate_candidates(
            base_state,
            current_state,
            decision
        )
    )

    if not candidates:
        print()
        print(
            "No candidate graph evolutions were generated."
        )

        return None

    print()

    print(
        f"Candidate Graphs Generated : "
        f"{len(candidates)}"
    )

    # STEP 4
    # EXTRACT PRIMARY CHANGES
    primary_changes = (
        _extract_primary_changes(
            decision
        )
    )

    print()

    print(
        f"Primary Changes : "
        f"{len(primary_changes)}"
    )

    for index, change in enumerate(
        primary_changes,
        start=1
    ):
        print(
            f"   {index}. "
            f"{change}"
        )

    # STEP 5
    # CANDIDATE RANKING
    print()

    print(
        "-" * 80
    )

    print(
        "STEP 3 : CANDIDATE RANKING"
    )

    print(
        "-" * 80
    )

    ranking_engine = (
        CandidateRankingEngine(

            weight_propagation=
                0.40,

            weight_change_consistency=
                0.30,

            weight_stability=
                0.20,

            weight_parsimony=
                0.10
        )
    )

    ranking_result = (
        ranking_engine.rank(
            candidates=
                candidates,

            base_state=
                base_state,

            primary_changes=
                primary_changes
        )
    )

    if not ranking_result.ranked_candidates:
        print()
        print(
            "No ranked candidates were produced."
        )

        return None

    excel_path = (
        ranking_engine.export_to_excel(
            ranking_result
        )
    )

    text_path = (
        ranking_engine.export_to_text(
            ranking_result,
            top_n=20
        )
    )

    # STEP 6
    # SELECT RANK 1
    rank_1 = (
        ranking_result.top_candidates
    )

    if not rank_1:
        raise RuntimeError(
            "Candidate ranking produced no Rank-1 candidate."
        )

    if len(rank_1) > 1:

        print()

        print(
            "=" * 80
        )

        print(
            "RANK-1 TIE ANALYSIS"
        )

        print(
            "=" * 80
        )

        print()

        print(
            f"Number of Rank-1 candidates : "
            f"{len(rank_1)}"
        )

        print()

        for tied_candidate in rank_1:

            print(
                "-" * 80
            )

            print(
                f"Rank : "
                f"{tied_candidate.rank}"
            )

            print(
                f"Candidate ID : "
                f"{tied_candidate.candidate_index}"
            )

            print(
                f"Final Score : "
                f"{tied_candidate.final_score:.6f}"
            )

            print(
                f"Propagation Plausibility : "
                f"{tied_candidate.propagation_plausibility:.6f}"
            )

            print(
                f"Change Consistency : "
                f"{tied_candidate.change_consistency:.6f}"
            )

            print(
                f"Graph Stability : "
                f"{tied_candidate.graph_stability:.6f}"
            )

            print(
                f"Parsimony : "
                f"{tied_candidate.parsimony:.6f}"
            )

            print(
                f"Structural Changes : "
                f"{tied_candidate.structural_changes}"
            )

            print(
                f"Topology Changes : "
                f"{tied_candidate.topology_changes}"
            )

            print(
                f"Weight Changes : "
                f"{tied_candidate.weight_changes}"
            )

            print(
                f"Interaction Plan : "
                f"{tied_candidate.interaction_plan}"
            )

            print(
                f"Interaction Level : "
                f"{tied_candidate.interaction_level}"
            )

            print(
                f"Propagation Rule : "
                f"{tied_candidate.propagation_rule}"
            )

            candidate_graph = (
                tied_candidate.candidate_state.graph
            )

            print()

            print(
                "Candidate Graph Edges:"
            )

            for edge in sorted(
                candidate_graph.edges(),
                key=str
            ):

                print(
                    f"   {edge}"
                )

            print()

        print(
            "=" * 80
        )

        print(
            "Tie analysis complete."
        )

        print(
            "No graph evolution has been committed."
        )

        print(
            "=" * 80
        )

        return {
            "ranking_result":
                ranking_result,

            "updated_state":
                None,

            "committed":
                False
        }
    # if len(
    #     rank_1
    # ) > 1:

    #     print()

    #     print(
    #         "WARNING:"
    #     )

    #     print(
    #         f"There are {len(rank_1)} "
    #         f"Rank-1 candidates."
    #     )

    #     print(
    #         "The graph update stage will NOT "
    #         "arbitrarily choose one."
    #     )

    #     print(
    #         "Resolve the Rank-1 tie before committing "
    #         "a graph evolution."
    #     )

        # return {
        #     "ranking_result":
        #         ranking_result,

        #     "updated_state":
        #         None,

        #     "committed":
        #         False
        # }

    selected_ranked = (
        rank_1[0]
    )

    selected_candidate = (
        selected_ranked.candidate_state
    )

    print()

    print(
        "-" * 80
    )

    print(
        "RANK-1 SELECTION"
    )

    print(
        "-" * 80
    )

    print()

    print(
        f"Rank : "
        f"{selected_ranked.rank}"
    )

    print(
        f"Original Candidate ID : "
        f"{selected_ranked.candidate_index}"
    )

    print(
        f"Final Score : "
        f"{selected_ranked.final_score:.6f}"
    )

    print()

    print(
        f"Interaction Plan : "
        f"{selected_ranked.interaction_plan}"
    )

    print(
        f"Propagation Rule : "
        f"{selected_ranked.propagation_rule}"
    )

    print()

    print(
        f"Propagation Plausibility : "
        f"{selected_ranked.propagation_plausibility:.6f}"
    )

    print(
        f"Change Consistency : "
        f"{selected_ranked.change_consistency:.6f}"
    )

    print(
        f"Graph Stability : "
        f"{selected_ranked.graph_stability:.6f}"
    )

    print(
        f"Parsimony : "
        f"{selected_ranked.parsimony:.6f}"
    )

    print()

    print(
        f"Structural Changes : "
        f"{selected_ranked.structural_changes}"
    )

    print(
        f"Topology Changes : "
        f"{selected_ranked.topology_changes}"
    )

    print(
        f"Weight Changes : "
        f"{selected_ranked.weight_changes}"
    )

    # STEP 7
    # APPLY RANK-1 EVOLUTION
    print()

    print(
        "-" * 80
    )

    print(
        "STEP 4 : APPLY RANK-1 EVOLUTION"
    )

    print(
        "-" * 80
    )

    update_engine = (
        GraphUpdateEngine()
    )

    updated_state = (
        update_engine.update(
            base_state=
                base_state,

            selected_candidate=
                selected_candidate
        )
    )

    # STEP 8
    # DISPLAY FINAL UPDATE
    update_engine.print_update_summary(

        base_state=
            base_state,

        updated_state=
            updated_state,

        rank=
            selected_ranked.rank,

        candidate_index=
            selected_ranked.candidate_index,

        final_score=
            selected_ranked.final_score,

        interaction_plan=
            selected_ranked.interaction_plan,

        propagation_rule=
            selected_ranked.propagation_rule
    )


    # ==========================================================
    # STEP 9
    # PERSIST RANK-1 STATE
    # ==========================================================

    print()

    print(
        "=" * 80
    )

    if not commit:

        print(
            "DRY RUN COMPLETE"
        )

        print(
            "=" * 80
        )

        print()

        print(
            "The Rank-1 evolution was applied to an "
            "in-memory copy of the original base state."
        )

        print(
            "The stored active graph has NOT been modified."
        )

        print()

        print(
            "Use:"
        )

        print(
            "    run_pipeline(commit=True)"
        )

        print(
            "to persist the Rank-1 evolution."
        )

        print()

        print(
            "Ranking Excel :"
        )

        print(
            f"  {excel_path}"
        )

        print()

        print(
            "Ranking Report :"
        )

        print(
            f"  {text_path}"
        )

        print()

        print(
            "=" * 80
        )

        return {

            "ranking_result":
                ranking_result,

            "selected_candidate":
                selected_candidate,

            "updated_state":
                updated_state,

            "committed":
                False,

            "ranking_excel":
                excel_path,

            "ranking_text":
                text_path
        }

    # ==========================================================
    # COMMIT MODE
    # ==========================================================

    print(
        "COMMIT MODE"
    )

    print()

    print(
        "The Rank-1 evolution will now replace the "
        "currently accepted active causal state."
    )

    print()

    summary = (
        update_engine.summarize_update(
            base_state,
            updated_state
        )
    )

    if (
        summary["TotalEvolutionOperations"]
        == 0
    ):
        raise RuntimeError(
            "Rank-1 candidate produced no graph "
            "evolution. Active graph will not be updated."
        )
    
    manager.update(
        updated_state
    )

    print()

    print(
        "Active graph successfully updated."
    )

    print()

    print(
        f"New Active Graph Edge Count : "
        f"{updated_state.graph.number_of_edges()}"
    )

    print()

    print(
        "=" * 80
    )

    print(
        "GRAPH UPDATE COMMITTED"
    )

    print(
        "=" * 80
    )

    return {

        "ranking_result":
            ranking_result,

        "selected_candidate":
            selected_candidate,

        "updated_state":
            updated_state,

        "committed":
            True,

        "ranking_excel":
            excel_path,

        "ranking_text":
            text_path
    }
    # # STEP 9
    # # DRY RUN
    # print()

    # print(
    #     "=" * 80
    # )

    # print(
    #     "DRY RUN COMPLETE"
    # )

    # print(
    #     "=" * 80
    # )

    # print()

    # print(
    #     "The Rank-1 evolution was applied to an in-memory copy "
    #     "of the original base state."
    # )

    # print(
    #     "The stored/base graph has NOT been modified."
    # )

    # print()

    # print(
    #     "Ranking Excel :"
    # )

    # print(
    #     f"  {excel_path}"
    # )

    # print()

    # print(
    #     "Ranking Report :"
    # )

    # print(
    #     f"  {text_path}"
    # )

    # print()

    # print(
    #     "No persistence operation has been performed."
    # )

    # print(
    #     "The update should be reviewed before committing "
    #     "the new state."
    # )

    # print()

    # print(
    #     "=" * 80
    # )

    # return {

    #     "ranking_result":
    #         ranking_result,

    #     "selected_candidate":
    #         selected_candidate,

    #     "updated_state":
    #         updated_state,

    #     "committed":
    #         False,

    #     "ranking_excel":
    #         excel_path,

    #     "ranking_text":
    #         text_path
    # }

def _extract_primary_changes(
    decision: Dict
) -> List[Dict]:
    """
    Extract primary changes from the Change Gate decision.

    The Change Gate returns a structure of the form:

        decision
            |
            +-- DetectedChanges
                    |
                    +-- AddedEdges
                    +-- RemovedEdges
                    +-- WeightChangedEdges
                    +-- IsolatedNodes

    This follows the same primary-change construction used by
    GraphEvolutionEngine.
    """

    if not isinstance(
        decision,
        dict
    ):
        raise TypeError(
            "Change Gate decision must be a dictionary."
        )

    detected_changes = (
        decision.get(
            "DetectedChanges",
            {}
        )
    )

    if not isinstance(
        detected_changes,
        dict
    ):
        raise TypeError(
            "decision['DetectedChanges'] must be a dictionary."
        )

    primary_changes = []

    # ----------------------------------------------------------
    # 1. EDGE ADDITIONS
    # ----------------------------------------------------------

    for edge in detected_changes.get(
        "AddedEdges",
        []
    ):

        primary_changes.append(
            {
                "type":
                    "EDGE_ADDITION",

                "edge":
                    edge
            }
        )

    # ----------------------------------------------------------
    # 2. EDGE REMOVALS
    # ----------------------------------------------------------

    for edge in detected_changes.get(
        "RemovedEdges",
        []
    ):

        primary_changes.append(
            {
                "type":
                    "EDGE_REMOVAL",

                "edge":
                    edge
            }
        )

    # ----------------------------------------------------------
    # 3. WEIGHT CHANGES
    # ----------------------------------------------------------

    for change in detected_changes.get(
        "WeightChangedEdges",
        []
    ):

        primary_changes.append(
            {
                "type":
                    "WEIGHT_CHANGE",

                "edge":
                    change["edge"],

                "old_weight":
                    change["old_weight"],

                "new_weight":
                    change["new_weight"]
            }
        )

    # ----------------------------------------------------------
    # 4. ISOLATED NODES
    # ----------------------------------------------------------

    for node in detected_changes.get(
        "IsolatedNodes",
        []
    ):

        primary_changes.append(
            {
                "type":
                    "ISOLATED_NODE",

                "node":
                    node
            }
        )

    return primary_changes


if __name__ == "__main__":

    run_pipeline(
        commit=False
    )