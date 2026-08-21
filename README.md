# Adaptive Regime-Aware Causal Stress Testing

## Overview

This project implements an adaptive causal stress-testing framework that
updates an existing causal graph when structural changes are detected in
newly observed financial data.

The framework is designed around the idea that a causal graph should not
be rebuilt from scratch every time new data becomes available.

Instead:

1. A base causal graph is maintained.
2. New data is compared against the information represented by the base graph.
3. A Change Gate detects meaningful structural changes.
4. Detected changes are converted into primary changes.
5. Propagation rules are generated for those changes.
6. The Interaction Engine generates possible interaction hypotheses.
7. The Graph Evolution Engine converts those hypotheses into candidate
   graph evolutions.
8. Candidate graphs are ranked using multiple criteria.
9. The highest-ranked evolution can subsequently be applied incrementally
   to the original base graph.

The important design principle is:

    Updated Base Graph
        =
    Original Base Graph
        +
    Selected Graph Evolution

The selected candidate graph is therefore not treated as a complete
replacement for the original causal graph.


---

# Current Architecture

The current implementation follows this general pipeline:

    Financial Data
          |
          v
    Causal Graph / Current State
          |
          v
    +------------------+
    |    Change Gate   |
    +------------------+
          |
          | Primary Changes
          v
    Propagation Rules
          |
          v
    +----------------------+
    | Interaction Engine   |
    +----------------------+
          |
          | Interaction Plans / Hypotheses
          v
    +-------------------------+
    | Graph Evolution Engine  |
    +-------------------------+
          |
          | Candidate Graphs
          v
    +-------------------------+
    | Candidate Ranking       |
    +-------------------------+
          |
          | Rank 1 Evolution
          v
    Graph Update Stage
          |
          v
    Updated Base Causal Graph


---

# Project Structure

The project is organized into source modules, storage modules,
graph-evolution modules, and generated results.

A simplified structure is:

    Adaptive_Regime_Aware_Causal_Stress_Testing/
    |
    +-- src/
    |   |
    |   +-- change_gate/
    |   |
    |   +-- graph_evolution/
    |   |
    |   +-- storage/
    |   |
    |   +-- ...
    |
    +-- results/
    |   |
    |   +-- graphs/
    |   |
    |   +-- reports/
    |
    +-- README.md
    +-- .gitignore


The exact repository may contain additional folders or support files
depending on the current development state.


---

# Source Code

## `src/`

The `src` directory contains the implementation of the causal stress-testing
framework.

The main components currently developed are:

- Change Gate
- Interaction Candidate representation
- Interaction Plan generation
- Interaction Engine
- Graph Evolution Engine
- Candidate Ranking Engine
- Storage and causal-state management


---

# Change Detection

## `src/change_gate/`

This directory contains the logic responsible for detecting meaningful
changes between the stored/base causal state and the newly observed
state.

### `change_gate.py`

`change_gate.py` implements the Change Gate.

Its role is to determine whether newly observed information represents a
meaningful change in the causal structure.

The Change Gate currently identifies categories such as:

- Added edges
- Removed edges
- Weight-changed edges
- Isolated nodes

The detected changes are represented as primary changes and passed to
the subsequent graph-evolution stages.

The Change Gate is therefore the transition point between:

    Newly observed state

and:

    Proposed causal-graph evolution


---

# Graph Evolution

## `src/graph_evolution/`

This directory contains the core logic for generating and evaluating
possible causal-graph evolutions.

The major components currently implemented are:

- Interaction candidates
- Interaction plans
- Interaction hypotheses
- Interaction engine
- Graph evolution engine
- Candidate ranking


---

## `interaction_candidate.py`

This module defines the `InteractionCandidate` data structure.

An interaction candidate represents a single action that may participate
in an interaction plan.

The candidate stores information such as:

- Action/operator
- Propagation rule
- Source primary-change index
- Edge associated with the action
- Node associated with the action
- Base score
- Additional metadata

The candidate representation is intentionally independent of direct
graph mutation.

This separation allows the framework to reason about possible actions
before actually modifying a graph.


---

## `interaction_plan.py`

This module represents an interaction plan.

An interaction plan describes a combination of actions that may jointly
explain or respond to detected structural changes.

The interaction-plan layer separates:

    What actions are being proposed?

from:

    How will those actions eventually modify the graph?

This is important because multiple plausible interaction plans may exist
for the same set of detected changes.


---

## `interaction_engine.py`

The Interaction Engine converts propagation rules into interaction
candidates and interaction hypotheses.

Its responsibilities include:

1. Building interaction candidates.
2. Preserving the relationship between a candidate and its source
   primary change.
3. Deduplicating equivalent candidates.
4. Generating level-one interaction hypotheses.
5. Generating higher-level interaction hypotheses.
6. Constructing interaction names.
7. Producing interaction plans for downstream graph evolution.

The engine supports multiple interaction levels.

The interaction engine does not directly replace the base causal graph.

Instead, it generates possible actions that can later be evaluated by the
Graph Evolution Engine.


---

## `graph_evolution_engine.py`

The Graph Evolution Engine converts interaction plans into candidate
graph evolutions.

Its role is to:

1. Receive the detected changes and interaction plans.
2. Apply possible graph-evolution operators.
3. Generate multiple plausible candidate graphs.
4. Assign a plausibility score to each candidate.
5. Preserve the relationship between the candidate graph and the
   interaction plan that produced it.

The generated candidates represent possible future/evolved states.

They are not automatically selected as the new base graph.

For example, an interaction plan may generate an evolution such as:

    EdgeAddition[A -> B]

or:

    EdgeRemoval[A -> B]

or:

    WeightChange[A -> B]

or an isolation/recovery operation.

The engine can therefore produce many candidate graphs for the same
detected set of changes.


---

# Operators and Propagation

## `operators.py`

The operators module contains the graph-evolution operations used by
the framework.

Examples of evolution operations include:

- Edge addition
- Edge removal
- Weight modification
- Recovery-related operations
- Other structural graph mutations used by the propagation rules

The operator layer is responsible for defining how an individual
evolution action can be represented/applied.

A major design constraint is that contradictory operations should not be
applied to the same edge simultaneously.

For example:

    AddEdge(A -> B)

and:

    RemoveEdge(A -> B)

should not occur together in the same valid graph-evolution candidate.


---

## `propagation_rules.py`

This module defines propagation rules associated with detected primary
changes.

Propagation rules describe how a detected structural change may
propagate through the causal system.

Examples include rules corresponding to:

- Keeping an added edge
- Applying a new edge weight
- Reducing downstream influence
- Recovery of isolated nodes
- Reconnection to strong causal parents/children

Propagation rules provide the basis from which the Interaction Engine
creates interaction candidates.


---

# Candidate Ranking

## `candidate_ranking.py`

The Candidate Ranking Engine evaluates all candidate graph evolutions
generated by the Graph Evolution Engine.

The initial implementation ranked candidates primarily using the
propagation plausibility score.

This resulted in a large number of tied Rank-1 candidates because many
candidate graphs inherited the same propagation score.

The ranking system was therefore expanded into a multi-criteria ranking
framework.

The current four criteria are:

### 1. Propagation Plausibility

Represents the plausibility associated with the propagation rule used
to generate the candidate.

This is currently based on the existing candidate score generated by
the graph-evolution pipeline.

It should be interpreted as a ranking component rather than a calibrated
probability.


### 2. Change Consistency

Measures whether the candidate actually implements the detected primary
change relative to the original base graph.

For example:

- An edge addition should introduce an edge that was absent from the
  base graph.
- An edge removal should remove an edge that existed in the base graph.
- A weight change should modify the weight of an existing edge.
- An isolation/recovery operation should introduce the appropriate
  connectivity.


### 3. Graph Stability

Measures how much of the original causal graph is preserved.

The current implementation considers both:

- Edge topology
- Edge weights

Topological similarity is evaluated using edge-set similarity.

Weight similarity is also incorporated so that a candidate that changes
edge weights is recognized as a modification even when the topology
remains unchanged.


### 4. Parsimony

Rewards smaller graph evolutions.

The current formulation treats both:

- Topological changes
- Edge-weight changes

as graph-evolution operations.

The parsimony score is:

    1 / (1 + number_of_evolution_operations)

Therefore:

    0 operations -> 1.0
    1 operation  -> 0.5
    2 operations -> 0.333...
    3 operations -> 0.25
    ...


---

# Final Ranking Score

The current ranking engine combines the four criteria using configurable
weights.

The default weights are:

    Propagation Plausibility : 0.40
    Change Consistency       : 0.30
    Graph Stability          : 0.20
    Parsimony                : 0.10

The final score is:

    Final Score =
        0.40 * Propagation Plausibility
        +
        0.30 * Change Consistency
        +
        0.20 * Graph Stability
        +
        0.10 * Parsimony

These weights are currently engineering/model-design parameters.

They are not intended to represent calibrated statistical probabilities.


---

# Ranking Output

The ranking module exports two result files.

## `results/graphs/candidate_ranking.xlsx`

Contains the complete ranking of candidate graphs.

The workbook contains:

### `Summary`

Provides:

- Total candidate count
- Highest final score
- Number of Rank-1 candidates
- Whether a Rank-1 tie exists
- Number of unique score levels
- Ranking weights
- Ranking method


### `Ranked Candidates`

Contains every candidate and its ranking information.

Important columns include:

- Rank
- Original Candidate ID
- Final Score
- Propagation Plausibility
- Change Consistency
- Graph Stability
- Parsimony
- Structural Changes
- Topology Changes
- Weight Changes
- Interaction Plan
- Interaction Level
- Propagation Rule
- Tie Group


### `Top Candidates`

Contains the candidates that received Rank 1.

This makes it easy to inspect whether a unique best candidate exists.


---

## `results/reports/candidate_ranking.txt`

Contains a compact human-readable version of the ranking results.

It includes:

- Candidate count
- Highest final score
- Number of Rank-1 candidates
- Ranking weights
- Top-ranked candidates
- Individual ranking criteria


---

# Important Design Principle: Candidate ID vs Rank

The ranking system preserves the original candidate index.

For example:

    Rank : 1
    Original Candidate ID : 40

does NOT mean that candidate 40 is the 40th-ranked candidate.

It means:

- Candidate 40 was generated as the 40th candidate during graph evolution.
- After ranking, it received Rank 1.

This distinction is intentional and allows a ranked candidate to be traced
back to its original graph-evolution output.


---

# Current Experimental Result

During development, the framework produced a large number of candidate
graphs.

An earlier ranking implementation produced thousands of candidates tied
at Rank 1 because candidates with the same propagation-rule score were
assigned the same final score.

The multi-criteria ranking approach was introduced to resolve this issue.

In the latest run, approximately 9,669 candidate graphs were generated.

The ranking produced:

    Highest Final Score : 0.949427
    Rank-1 Candidates    : 1
    Top-Rank Tie         : No

The highest-ranked candidate was:

    WeightChange[
        SP500_Return
        ->
        IndustrialProductionGrowth
    ]

with:

    Propagation Plausibility : 1.0000
    Change Consistency       : 1.0000
    Graph Stability          : 0.9971
    Parsimony                : 0.5000

The candidate represented:

    Topology Changes : 0
    Weight Changes   : 1
    Structural Changes : 1

This is an important result because it demonstrates that the framework
can select a graph evolution that modifies an existing causal
relationship without unnecessarily changing the graph topology.


---

# Storage

## `src/storage/`

The storage directory contains components responsible for maintaining
causal states and the currently active/base graph.

### `active_graph_manager.py`

Responsible for managing the active/stored causal graph state.

The stored state represents the existing base graph against which newly
observed states are evaluated.

The base graph is an important reference throughout the pipeline.

The ranking stage compares candidates against this original base state.


### `causal_state_repository.py`

Responsible for loading and managing stored causal states.

The repository provides the current states used by downstream processing.


---

# Results Directory

## `results/`

The `results` directory contains generated outputs rather than source
code.

It is divided into categories such as:

    results/
    |
    +-- graphs/
    |
    +-- reports/

### `results/graphs/`

Contains graph-related generated outputs.

Currently this includes:

    candidate_ranking.xlsx


### `results/reports/`

Contains human-readable reports.

Currently this includes:

    candidate_ranking.txt


The results directory contains generated artifacts and is therefore
excluded from version control through `.gitignore`.


---

# Current End-to-End Workflow

The current implementation can be understood as the following sequence.

## Step 1 — Maintain the Base Graph

The system maintains an existing causal graph representing the current
base state.

This graph is not immediately replaced when new data arrives.


## Step 2 — Observe New Data

A new/current causal state is obtained from the incoming data.


## Step 3 — Change Detection

The Change Gate compares the stored/base state with the current state.

It detects meaningful changes such as:

- Edge additions
- Edge removals
- Weight changes
- Isolated nodes


## Step 4 — Primary Changes

Detected changes are represented as primary changes.

These become the input to the propagation and interaction stages.


## Step 5 — Propagation Rules

Propagation rules are generated for the detected changes.

These describe possible causal responses to the structural changes.


## Step 6 — Interaction Candidates

The Interaction Engine converts propagation rules into normalized
interaction candidates.

Candidates retain their source-change information so that interactions
can be traced back to the changes that produced them.


## Step 7 — Interaction Plans

Multiple actions may be combined into interaction hypotheses and plans.

The system therefore considers multiple plausible causal responses.


## Step 8 — Candidate Graph Generation

The Graph Evolution Engine applies valid graph-evolution operators to
interaction plans and generates candidate graph states.

The candidate graphs represent possible explanations/evolutions of the
base causal graph.


## Step 9 — Candidate Ranking

All candidate graphs are evaluated using:

- Propagation Plausibility
- Change Consistency
- Graph Stability
- Parsimony

The candidates are then ranked.


## Step 10 — Rank-1 Selection

The highest-ranked candidate represents the currently preferred graph
evolution.

This is where the current implementation stops.

The next stage is to apply the selected evolution to the original base
graph.


---

# Next Planned Stage

The next component to implement is the graph-update stage.

Its purpose will be:

    Original Base Graph
            +
    Rank-1 Graph Evolution
            |
            v
    Updated Base Graph

The important distinction is:

    Candidate Graph != Replacement Base Graph

Instead, the Rank-1 candidate describes the evolution that should be
applied to the existing base graph.

For example, if Rank 1 is:

    WeightChange[
        SP500_Return
        ->
        IndustrialProductionGrowth
    ]

then the update stage should:

1. Load the original base graph.
2. Identify the affected edge.
3. Apply the selected new weight.
4. Preserve all unrelated edges and graph structure.
5. Produce the updated base graph.
6. Store the updated causal state.

The update stage should not rebuild or replace the entire graph.


---

# Design Philosophy

The framework is designed around several principles.

## Incremental Graph Evolution

The causal graph evolves incrementally rather than being completely
reconstructed whenever new information appears.


## Multiple Plausible Evolutions

The system does not immediately commit to the first possible graph.

It generates multiple candidate evolutions and ranks them.


## Traceability

Every candidate should be traceable back through:

    Candidate Graph
        |
        v
    Interaction Plan
        |
        v
    Propagation Rule
        |
        v
    Primary Change
        |
        v
    Change Gate


## Separation of Responsibilities

The system separates:

- Change detection
- Propagation reasoning
- Interaction generation
- Graph evolution
- Candidate ranking
- Final graph updating

This prevents the ranking system from directly mutating the graph and
makes the framework easier to test and extend.


## Conservative Evolution

The ranking criteria favour candidates that explain detected changes while
preserving as much of the existing causal structure as possible.

This is particularly important for financial causal graphs, where
unnecessary structural changes can produce unstable or difficult-to-
interpret models.


---

# Current Status

Implemented:

- Change Gate
- Primary change representation
- Propagation rules
- Interaction candidates
- Interaction plans
- Interaction Engine
- Graph Evolution Engine
- Candidate graph generation
- Multi-criteria candidate ranking
- Excel ranking export
- Text ranking export

Current ranking criteria:

- Propagation Plausibility
- Change Consistency
- Graph Stability
- Parsimony

Current result:

- Multiple candidate graph evolutions generated
- Candidate ranking implemented
- Unique Rank-1 candidate obtained in the latest run

Next implementation stage:

- Rank-1 graph-evolution application
- Incremental base-graph update
- Updated causal-state persistence


---

# Version Control

Generated results are intentionally excluded from Git.

The `.gitignore` contains:

    results/

This prevents generated Excel files, reports, and other experimental
outputs from being committed to the repository.

Source code, configuration, documentation, and other project files remain
tracked normally.


---

# Development Notes

The ranking weights are currently provisional and should be treated as
part of the experimental design.

Future improvements may include:

- Empirical calibration of ranking weights
- Additional structural consistency metrics
- Better edge-weight similarity measures
- Regime-aware ranking
- Historical validation of candidate selection
- Out-of-sample validation
- Comparison against alternative causal graph update strategies
- Machine-learning-based Change Gate detection
- More sophisticated probabilistic candidate ranking


---

# Summary

The framework currently implements an adaptive causal graph evolution
pipeline:

    Detect
       ↓
    Explain
       ↓
    Generate
       ↓
    Rank
       ↓
    Update

The key distinction is that the system does not replace the existing
causal graph with an independently generated graph.

Instead, it identifies the most plausible evolution and applies that
evolution incrementally to the existing base graph.

The next stage is therefore the implementation of the graph-update
mechanism that applies the Rank-1 evolution to the original base graph.