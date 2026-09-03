import json
import sys
from pathlib import Path
from typing import List

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.core.causal_state import CausalState
from src.change_gate.change_gate import ChangeGate
from src.graph_evolution.graph_evolution_engine import GraphEvolutionEngine
from src.graph_evolution.candidate_ranking import CandidateRankingEngine

RESULTS_DIR = ROOT / "results"
CAUSAL_STATES_DIR = RESULTS_DIR / "causal_states"
ACTIVE_GRAPH_PATH = RESULTS_DIR / "active_graph" / "active_state.pkl"
PRESENTATION_SCENARIO_PATH = ROOT / "presentation_samples" / "market_stress_graph_change.json"


@st.cache_data
def list_saved_states() -> List[int]:
    if not CAUSAL_STATES_DIR.exists():
        return []

    ids = []
    for file in CAUSAL_STATES_DIR.glob("causal_state_*.pkl"):
        try:
            ids.append(int(file.stem.split("_")[-1]))
        except ValueError:
            continue

    return sorted(ids)


@st.cache_data
def load_state(window_id: int):
    file_path = CAUSAL_STATES_DIR / f"causal_state_{window_id:04d}.pkl"
    if file_path.exists():
        return joblib.load(file_path)
    raise FileNotFoundError(f"State for window {window_id} not found")


@st.cache_data
def load_active_state():
    if ACTIVE_GRAPH_PATH.exists():
        return joblib.load(ACTIVE_GRAPH_PATH)
    return None


def build_demo_state(window_id: int = 0) -> CausalState:
    graph = nx.DiGraph()
    nodes = [
        "GDP",
        "Inflation",
        "PolicyRate",
        "CreditSpread",
        "EquityIndex",
        "Volatility",
        "CommodityPrice",
        "Liquidity",
    ]
    graph.add_nodes_from(nodes)

    edges = [
        ("GDP", "Inflation", 0.72),
        ("PolicyRate", "Inflation", -0.64),
        ("Inflation", "CreditSpread", 0.81),
        ("CreditSpread", "EquityIndex", -0.69),
        ("Volatility", "EquityIndex", -0.74),
        ("CommodityPrice", "Inflation", 0.58),
        ("Liquidity", "EquityIndex", 0.66),
        ("PolicyRate", "Liquidity", -0.54),
        ("GDP", "Liquidity", 0.49),
        ("Inflation", "Volatility", 0.61),
    ]

    graph.add_weighted_edges_from(edges)

    n = len(nodes)
    adj = np.zeros((n, n), dtype=float)
    for i, src in enumerate(nodes):
        for j, dst in enumerate(nodes):
            if graph.has_edge(src, dst):
                adj[i, j] = graph[src][dst].get("weight", 0.0)

    state = CausalState(
        window_id=window_id,
        start_date="2024-01-01",
        end_date="2024-01-30",
        dominant_regime=1,
        regime_distribution={0: 0.26, 1: 0.74},
        average_confidence=0.82,
        average_entropy=0.43,
        graph=graph,
        adjacency_matrix=adj,
        edge_strength_matrix=adj.copy(),
        p_value_matrix=np.eye(n, dtype=float),
        variable_names=nodes,
    )

    state.graph_metrics = {
        "density": round(nx.density(graph), 3),
        "average_degree": round(sum(dict(graph.degree()).values()) / n, 3),
        "num_edges": graph.number_of_edges(),
    }

    state.change_features = {
        "RegimeChanged": False,
        "SHD": 2,
        "EntropyDifference": 0.12,
        "Jaccard": 0.89,
    }

    return state


def build_presentation_scenario():
    """Load an isolated synthetic stress scenario for dashboard demonstrations."""
    with PRESENTATION_SCENARIO_PATH.open(encoding="utf-8") as file:
        scenario = json.load(file)

    base_state = build_demo_state(scenario["base_window_id"])
    observed_state = build_demo_state(scenario["observed_window_id"])
    changes = scenario["changes"]

    removed_source, removed_target = changes["remove_edge"]
    observed_state.graph.remove_edge(removed_source, removed_target)

    added_source, added_target, added_weight = changes["add_edge"]
    observed_state.graph.add_edge(added_source, added_target, weight=added_weight)

    weight_source, weight_target, new_weight = changes["change_weight"]
    observed_state.graph[weight_source][weight_target]["weight"] = new_weight
    observed_state.dominant_regime = 2
    observed_state.average_confidence = 0.67
    observed_state.average_entropy = 0.69
    observed_state.change_features = {
        "RegimeChanged": True,
        "SHD": 2,
        "EntropyDifference": 0.26,
        "Jaccard": 0.82,
    }
    return base_state, observed_state, scenario


@st.cache_data
def get_available_states():
    return list_saved_states()


def draw_graph(graph: nx.DiGraph, title: str, highlight_edges=None):
    if graph is None or graph.number_of_nodes() == 0:
        st.info(f"No graph available for {title}.")
        return

    pos = nx.spring_layout(graph, seed=42)

    edge_colors = []
    edge_widths = []
    for u, v in graph.edges():
        if highlight_edges is not None and (u, v) in highlight_edges:
            edge_colors.append("#ff7f0e")
            edge_widths.append(2.7)
        else:
            edge_colors.append("#4c78a8")
            edge_widths.append(1.8)

    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw_networkx_nodes(graph, pos, node_color="#d8e6ff", node_size=1100, edgecolors="#1f1f1f", ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=11, ax=ax)
    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.9,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, pad=18)
    ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig)


def summarize_graph(state: CausalState):
    graph = state.graph
    if graph is None:
        return {}

    return {
        "Nodes": graph.number_of_nodes(),
        "Edges": graph.number_of_edges(),
        "Density": round(nx.density(graph), 4),
        "AvgDegree": round(sum(dict(graph.degree()).values()) / max(1, graph.number_of_nodes()), 4),
        "DominantRegime": state.dominant_regime,
        "AvgConfidence": round(float(state.average_confidence), 4),
        "AvgEntropy": round(float(state.average_entropy), 4),
    }


def diff_graph(previous: CausalState, current: CausalState):
    prev_graph = previous.graph
    curr_graph = current.graph

    merged = nx.DiGraph()
    nodes = set(prev_graph.nodes()) | set(curr_graph.nodes())
    merged.add_nodes_from(nodes)

    for u, v in curr_graph.edges():
        if not prev_graph.has_edge(u, v):
            merged.add_edge(u, v, kind="added")
        else:
            prev_weight = prev_graph[u][v].get("weight", 0.0)
            curr_weight = curr_graph[u][v].get("weight", 0.0)
            if abs(prev_weight - curr_weight) > 1e-6:
                merged.add_edge(u, v, kind="changed")

    for u, v in prev_graph.edges():
        if not curr_graph.has_edge(u, v):
            merged.add_edge(u, v, kind="removed")

    return merged


def primary_changes_from_decision(decision):
    detected = decision.get("DetectedChanges", {})
    changes = []

    for edge in detected.get("AddedEdges", []):
        changes.append({"type": "EDGE_ADDITION", "edge": edge})
    for edge in detected.get("RemovedEdges", []):
        changes.append({"type": "EDGE_REMOVAL", "edge": edge})
    for change in detected.get("WeightChangedEdges", []):
        changes.append({"type": "WEIGHT_CHANGE", **change})
    for node in detected.get("IsolatedNodes", []):
        changes.append({"type": "ISOLATED_NODE", "node": node})

    return changes


def run_real_analysis(base_state, current_state):
    gate = ChangeGate()
    decision = gate.evaluate(base_state, current_state)
    primary_changes = primary_changes_from_decision(decision)

    if not decision["ShouldUpdate"]:
        return {"decision": decision, "candidates": [], "ranking": None}

    candidates = GraphEvolutionEngine().generate_candidates(
        base_state,
        current_state,
        decision,
    )
    ranking = CandidateRankingEngine().rank(
        candidates,
        base_state,
        primary_changes,
    )
    return {"decision": decision, "candidates": candidates, "ranking": ranking}


def demo_analysis(base_state, current_state):
    decision = {
        "GraphComparison": f"StoredGraph({base_state.window_id}) -> PCMCI({current_state.window_id})",
        "StructuralChangeScore": 0.86,
        "AdaptiveThreshold": 0.70,
        "ShouldUpdate": True,
        "DecisionConfidence": "HIGH",
        "DetectedChanges": {
            "AddedEdges": [("CreditSpread", "Volatility")],
            "RemovedEdges": [("GDP", "Liquidity")],
            "WeightChangedEdges": [{
                "edge": ("Inflation", "CreditSpread"),
                "old_weight": 0.81,
                "new_weight": 0.94,
            }],
            "IsolatedNodes": [],
            "PrimaryChangeCount": 3,
        },
        "PrimaryChangeCount": 3,
        "Reasons": [
            "Regime transition detected.",
            "Moderate structural change detected.",
            "Structural Change Score (0.860) exceeds adaptive threshold (0.700).",
        ],
        "FeatureVector": current_state.change_features,
    }
    evolved = build_demo_state(current_state.window_id)
    evolved.graph.remove_edge("GDP", "Liquidity")
    evolved.graph.add_edge("CreditSpread", "Volatility", weight=0.62)
    evolved.graph["Inflation"]["CreditSpread"]["weight"] = 0.94
    ranking = CandidateRankingEngine().rank(
        [evolved],
        base_state,
        primary_changes_from_decision(decision),
    )
    return {"decision": decision, "candidates": [evolved], "ranking": ranking}


def render_graph_panel(title: str, state: CausalState):
    st.subheader(title)
    metrics = summarize_graph(state)
    cols = st.columns(6)
    labels = ["Nodes", "Edges", "Density", "AvgDegree", "Regime", "Confidence"]
    values = [
        metrics.get("Nodes", 0),
        metrics.get("Edges", 0),
        metrics.get("Density", 0.0),
        metrics.get("AvgDegree", 0.0),
        metrics.get("DominantRegime", 0),
        metrics.get("AvgConfidence", 0.0),
    ]

    for col, label, value in zip(cols, labels, values):
        col.metric(label, value)

    draw_graph(state.graph, title)


def render_candidate_panel(analysis):
    st.subheader("Candidate graph ranking")
    ranking = analysis.get("ranking")
    if ranking is None or not ranking.ranked_candidates:
        st.info("No candidate graphs were generated because the Change Gate rejected the update.")
        return

    rows = []
    for item in ranking.ranked_candidates:
        rows.append({
            "Rank": item.rank,
            "Candidate": item.candidate_index,
            "Final score": round(item.final_score, 4),
            "Propagation": round(item.propagation_plausibility, 4),
            "Consistency": round(item.change_consistency, 4),
            "Stability": round(item.graph_stability, 4),
            "Parsimony": round(item.parsimony, 4),
            "Graph changes": item.structural_changes,
            "Top rank": item.is_top_rank,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    top = ranking.top_candidates[0]
    st.success(f"Selected candidate: #{top.candidate_index} with final score {top.final_score:.4f}")
    render_graph_panel("Selected evolved graph", top.candidate_state)


def main():
    st.set_page_config(page_title="Adaptive Causal Stress Testing Dashboard", layout="wide")
    st.title("Adaptive Causal Stress Testing Dashboard")
    st.caption("Regime-aware causal graph monitoring, structural change detection, and graph evolution review")

    with st.sidebar:
        st.header("Controls")
        state_ids = get_available_states()
        source = st.radio(
            "Visualization source",
            ["Existing causal states", "Presentation sample: market stress"],
        )

        if source == "Existing causal states" and state_ids:
            selected_window = st.selectbox("Select window", state_ids)
        elif source == "Existing causal states":
            selected_window = 0
            st.info("No saved states were found. The app is showing a demo graph instance.")
        else:
            selected_window = 9001
            st.success("Synthetic scenario loaded from presentation_samples/.")

        st.markdown("---")
        st.write("Saved artifact folders")
        st.write("- results/causal_states")
        st.write("- results/active_graph")
        st.write("- results/graphs")
        st.write("- results/reports")

    presentation_scenario = None
    if source == "Presentation sample: market stress":
        base_state, current_state, presentation_scenario = build_presentation_scenario()
        previous_state = base_state
    elif not state_ids:
        current_state = build_demo_state(selected_window)
        previous_state = build_demo_state(selected_window - 1)
        current_state.graph.remove_edge("GDP", "Liquidity")
        current_state.graph.add_edge("CreditSpread", "Volatility", weight=0.57)
        current_state.graph["Inflation"]["CreditSpread"]["weight"] = 0.91
    else:
        if selected_window not in state_ids:
            selected_window = state_ids[0]
        current_state = load_state(selected_window)

        if len(state_ids) > 1:
            prev_window = max(w for w in state_ids if w < selected_window) if any(w < selected_window for w in state_ids) else state_ids[0]
            previous_state = load_state(prev_window)
        else:
            previous_state = current_state

    has_real_states = bool(list_saved_states()) and source == "Existing causal states"
    active_state = load_active_state() if ACTIVE_GRAPH_PATH.exists() and has_real_states else previous_state
    base_state = active_state or previous_state

    if source == "Presentation sample: market stress":
        analysis = demo_analysis(base_state, current_state)
        analysis_source = presentation_scenario["scenario_name"]
    elif has_real_states:
        try:
            analysis = run_real_analysis(base_state, current_state)
            analysis_source = "real pipeline"
        except Exception as error:
            analysis = None
            analysis_source = "unavailable"
            st.error(f"Analysis could not run: {error}")
    else:
        analysis = demo_analysis(base_state, current_state)
        analysis_source = "demo data"

    summary = summarize_graph(current_state)
    st.subheader("Overview")
    st.info(f"Visualization source: {analysis_source}. Base graph: window {base_state.window_id}; observed graph: window {current_state.window_id}.")
    overview_cols = st.columns(6)
    overview_values = [
        summary.get("Nodes", 0),
        summary.get("Edges", 0),
        summary.get("Density", 0.0),
        summary.get("AvgDegree", 0.0),
        summary.get("DominantRegime", 0),
        f"{summary.get('AvgConfidence', 0.0):.3f}",
    ]
    overview_labels = ["Nodes", "Edges", "Density", "AvgDegree", "Regime", "Confidence"]

    for col, label, value in zip(overview_cols, overview_labels, overview_values):
        col.metric(label, value)

    st.markdown("---")

    tab_graph, tab_compare, tab_candidate, tab_status = st.tabs([
        "Current Graph",
        "Graph Comparison",
        "Candidate Ranking",
        "System Status",
    ])

    with tab_graph:
        render_graph_panel("Current causal graph", current_state)

    with tab_compare:
        st.subheader("Before, observed, and after")
        before_column, observed_column, after_column = st.columns(3)
        with before_column:
            st.write(f"Base / accepted · window {base_state.window_id}")
            draw_graph(base_state.graph, "Base graph")
        with observed_column:
            st.write(f"New observed · window {current_state.window_id}")
            draw_graph(current_state.graph, "Observed graph")
        with after_column:
            if analysis and analysis["ranking"] and analysis["ranking"].top_candidates:
                selected_state = analysis["ranking"].top_candidates[0].candidate_state
                st.write("Selected evolved graph")
                draw_graph(selected_state.graph, "Updated graph")
            else:
                st.info("No updated graph: the Change Gate rejected the observed change.")

        diff = diff_graph(base_state, current_state)
        st.subheader("Detected change summary")
        added = sum(1 for _, _, data in diff.edges(data=True) if data.get("kind") == "added")
        removed = sum(1 for _, _, data in diff.edges(data=True) if data.get("kind") == "removed")
        changed = sum(1 for _, _, data in diff.edges(data=True) if data.get("kind") == "changed")
        st.json({"added_edges": added, "removed_edges": removed, "weight_changed_edges": changed})

        if analysis:
            detected = analysis["decision"].get("DetectedChanges", {})
            change_rows = []
            for key, label in [("AddedEdges", "Added edge"), ("RemovedEdges", "Removed edge"), ("WeightChangedEdges", "Weight changed"), ("IsolatedNodes", "Isolated node")]:
                for value in detected.get(key, []):
                    change_rows.append({"Type": label, "Details": str(value)})
            if change_rows:
                st.dataframe(pd.DataFrame(change_rows), use_container_width=True, hide_index=True)

    with tab_candidate:
        if analysis:
            render_candidate_panel(analysis)
        else:
            st.info("Candidate ranking is unavailable until the analysis completes.")

    with tab_status:
        if analysis:
            decision = analysis["decision"]
            st.subheader("Change Gate decision")
            decision_cols = st.columns(4)
            decision_cols[0].metric("Change score", f"{decision['StructuralChangeScore']:.4f}")
            decision_cols[1].metric("Adaptive threshold", f"{decision['AdaptiveThreshold']:.4f}")
            decision_cols[2].metric("Primary changes", decision["PrimaryChangeCount"])
            decision_cols[3].metric("Decision confidence", decision["DecisionConfidence"])
            if decision["ShouldUpdate"]:
                st.success("Update accepted: candidate graph evolution was generated.")
            else:
                st.warning("Update rejected: the current graph remains the accepted graph.")
            st.subheader("Why the gate decided this")
            for reason in decision.get("Reasons", []):
                st.write(f"- {reason}")

        st.subheader("Repository status")
        state_ids = list_saved_states()
        st.write(f"Saved causal states found: {len(state_ids)}")
        if state_ids:
            st.dataframe(pd.DataFrame({"window_id": state_ids}), use_container_width=True)
        else:
            st.info("No saved windows exist yet. Generate them with the project pipeline or use the demo view.")

        st.write("Active graph status")
        st.write("Exists" if ACTIVE_GRAPH_PATH.exists() else "No active graph file found")

        if active_state is not None:
            st.json(
                {
                    "window_id": active_state.window_id,
                    "dominant_regime": active_state.dominant_regime,
                    "average_confidence": float(active_state.average_confidence),
                    "average_entropy": float(active_state.average_entropy),
                }
            )


if __name__ == "__main__":
    main()
