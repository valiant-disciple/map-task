# Paper 2: Spatial Knowledge Graphs from Collaborative Dialogue — Measuring Shared Mental Model Structure in Real Time

## Target Venue
Cognitive Science / Journal of Experimental Psychology: Applied / Frontiers in Psychology

---

## 1. Central Thesis

Shared mental models (SMMs) in collaborative spatial tasks are traditionally measured via post-hoc self-report (surveys) or offline structural tasks (card sorts, concept maps). We propose extracting **spatial knowledge graphs** directly from dialogue in real time using LLM-based analysis, and show that: (a) dialogue-derived graph similarity between partners predicts task performance better than self-reported perceived SMM, (b) graph structure evolves within trials as partners build common ground, and (c) the gap between dialogue-derived structure and ground-truth map structure reveals specific failure modes (referential ambiguity, frame misalignment).

---

## 2. Research Questions

**RQ1**: Can spatial knowledge graphs extracted from collaborative dialogue via LLMs capture meaningful structural differences between dyads?

**RQ2**: Does dialogue-derived KG similarity between Director and Matcher predict task performance (IoU, pass/fail) above and beyond self-reported PSMM?

**RQ3**: How does spatial knowledge graph structure evolve within a trial as partners build common ground?

**RQ4**: What structural properties of dialogue-derived KGs distinguish successful from unsuccessful trials?

**RQ5**: Does KG similarity mediate the relationship between communication quality and task performance?

---

## 3. Hypotheses

**H1**: Dyads with higher KG similarity (more aligned spatial mental models) achieve higher drawing accuracy (IoU, boundary F1) and faster completion.

**H2**: Dialogue-derived KG similarity explains variance in performance beyond what PSMM self-report captures — because it reflects actual information exchange rather than perceived alignment.

**H3**: KG structure becomes more similar between partners over the course of a trial (convergence), and the rate of convergence predicts final accuracy.

**H4**: Failing trials show:
- (a) Lower landmark agreement (fewer shared nodes)
- (b) More spatial frame misalignment (egocentric vs. allocentric references)
- (c) Higher proportion of REPAIR dialogue acts
- (d) Greater divergence between dialogue KG and ground truth KG

**H5**: Communication quality (specificity, grounding, spatial precision from LLM eval) → KG alignment → Performance, with KG alignment mediating.

---

## 4. Knowledge Graph Construction

### 4.1 Vision: Landmark Extraction from Map Images

**Input**: Map GIF files (`map{N}f.gif` for Director, `map{N}g.gif` for Matcher)

**Method**: GPT-4.1 vision API with structured JSON output

**Extracted per landmark**:
- `name`: landmark label
- `type`: building, tree, lake, bridge, etc.
- `x_pct`, `y_pct`: position as percentage of map dimensions
- `size`: small / medium / large
- `has_route`: whether the gold route passes near/through it (Director map only)
- `route_description`: spatial relationship to the route

**Caching**: Results cached per map number in `.landmark_cache/` to avoid redundant API calls.

**What exists**: Fully implemented in `knowledge_graph.py`. Vision extraction, caching, JSON parsing all working.

### 4.2 Text: Spatial Relation Extraction from Dialogue

**Input**: ASR transcripts from Director and Matcher (from Whisper or Smallest Pulse)

**Method**: GPT-4.1 text API with structured JSON output

**Extracted per trial**:
- `relations[]`: list of spatial relations
  - `source`: landmark name
  - `target`: landmark name
  - `relation_type`: NEAR, LEFT_OF, RIGHT_OF, ABOVE, BELOW, BETWEEN, ALONG, PAST, AROUND, THROUGH
  - `confidence`: 0.0–1.0
  - `speaker`: director / matcher
  - `context`: surrounding utterance
- `route_sequence[]`: ordered list of landmarks mentioned in route order
- `landmarks_mentioned[]`: all landmarks referenced in dialogue
- `spatial_frame`: egocentric / allocentric / mixed
- `reference_frame_switches`: count of switches between frames

**What exists**: Fully implemented in `knowledge_graph.py`. Extraction, parsing, JSON schema all working.

### 4.3 Graph Construction

**Method**: NetworkX directed graph

- **Nodes** = landmarks (from vision extraction + dialogue mentions)
- **Edges** = spatial relations (from dialogue extraction)
  - Edge attributes: `relation_type`, `confidence`, `speaker`, `count` (repeated mentions)
- **Node attributes**: `x_pct`, `y_pct`, `mentioned_by` (director/matcher/both), `on_route` (boolean)

**What exists**: NetworkX graph construction in `knowledge_graph.py`. Basic graph building works.

**What needs building**:
- Separate graphs per speaker (Director KG vs. Matcher KG)
- Ground truth graph from GT JSON route files
- Graph comparison metrics (see Section 5)

---

## 5. Graph Comparison Metrics

### 5.1 Between Director and Matcher (SMM Alignment)

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Node Jaccard | \|V_D ∩ V_M\| / \|V_D ∪ V_M\| | Shared landmark vocabulary |
| Edge Jaccard | \|E_D ∩ E_M\| / \|E_D ∪ E_M\| (ignoring relation types) | Shared spatial relation structure |
| Typed Edge Jaccard | Same but relation type must match | Strict structural agreement |
| Route Sequence Edit Distance | Levenshtein distance between route orderings | Sequential agreement on path |
| Route Sequence Kendall τ | Rank correlation of shared landmark positions in route | Monotonic agreement |
| Spatial Frame Agreement | Both use same frame (ego/allo) | Reference frame alignment |
| Graph Edit Distance (GED) | Min edits (node/edge add/remove/relabel) to transform G_D → G_M | Overall structural dissimilarity |
| QAP Correlation | Correlation of adjacency matrices (Quadratic Assignment Procedure) | Classic SMM similarity from Pathfinder tradition |

### 5.2 Between Dialogue KG and Ground Truth

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Landmark Recall | Proportion of GT route landmarks mentioned | Coverage of critical landmarks |
| Landmark Precision | Proportion of mentioned landmarks that are on GT route | Focus on relevant landmarks |
| Route Order Accuracy | Edit distance between mentioned route and GT route | Sequential correctness of instructions |
| Spatial Relation Accuracy | Proportion of extracted relations consistent with GT map geometry | Factual correctness of spatial descriptions |

### 5.3 What Needs Building

| Component | Status | Effort |
|-----------|--------|--------|
| Per-speaker graph construction | Partially done (single combined graph) | Low — split by `speaker` attribute |
| GT graph from route JSON | Not built | Medium — parse GT coordinates, compute spatial relations between route landmarks |
| Node/Edge Jaccard | Not built | Low |
| Route sequence edit distance | Not built | Low (Levenshtein) |
| GED | Not built | Medium (NX has `graph_edit_distance()` but slow; approximate needed for larger graphs) |
| QAP | Not built | Medium (permutation-based matrix correlation) |
| Landmark recall/precision vs GT | Not built | Low |
| Temporal windowed KG (Section 6) | Not built | Medium |

---

## 6. Temporal Evolution of Knowledge Graphs

### 6.1 Approach

Split each trial's transcript into temporal windows (e.g., 60s windows with 30s overlap) and extract KGs per window. Track:

- **Landmark accumulation**: number of unique landmarks mentioned over time
- **Edge density**: relations / possible relations over time
- **Director–Matcher Jaccard over time**: does similarity increase?
- **Convergence rate**: slope of Jaccard vs. time
- **Convergence timing**: when does KG similarity plateau?

### 6.2 Expected Pattern

```
Time →   0s          60s         120s        180s        210s
         |            |            |            |           |
KG Sim:  0.1 ———→ 0.3 ———→ 0.5 ———→ 0.6       (successful trial)
KG Sim:  0.1 ———→ 0.2 ———→ 0.15 ——→ 0.2       (failing trial — no convergence)
```

### 6.3 What Needs Building

1. **Windowed transcript splitting**: segment ASR output by timestamp into 60s windows
2. **Per-window KG extraction**: run GPT extraction on each window separately
3. **Cumulative KG**: union of all windows up to time t
4. **Similarity time series**: Jaccard, edge overlap, route agreement at each window
5. **Convergence statistics**: slope, plateau detection, final similarity

---

## 7. Dialogue Analysis Integration

### 7.1 Dialogue Acts (from llm_eval.py)

10 categories: INSTRUCT, DESCRIBE, CHECK, QUERY, CLARIFY, ACKNOWLEDGE, REPAIR, META, FILLER, OTHER

**Predictions**:
- Trials with more INSTRUCT + DESCRIBE acts have richer KGs (more nodes and edges)
- Trials with more REPAIR acts have lower KG similarity (misalignment being corrected)
- Higher CHECK/ACKNOWLEDGE rate → faster convergence (grounding loop)

### 7.2 Communication Quality (from llm_eval.py)

8 dimensions (1–7): clarity, specificity, efficiency, grounding, adaptiveness, spatial_precision, collaboration, overall_score

**Predictions**:
- Spatial precision → higher spatial relation accuracy in KG
- Grounding → higher Director–Matcher KG Jaccard
- Efficiency → fewer redundant edges, shorter route sequence

### 7.3 Linguistic Convergence (from llm_eval.py)

5 dimensions: lexical, syntactic, spatial frame alignment, conceptual, landmark agreement

**Predictions**:
- Landmark agreement → higher node Jaccard
- Spatial frame alignment → fewer frame switches, higher relation accuracy
- Lexical convergence → shared vocabulary reflects shared KG structure

**What exists**: All LLM eval features fully implemented in `llm_eval.py`.

---

## 8. Relation to Perceived SMM (PSMM Survey)

### 8.1 Validation Question

Does the dialogue-derived KG similarity correlate with PSMM self-report? If yes, it validates the KG method. If KG similarity predicts performance *beyond* PSMM, it demonstrates added value of behavioral measurement over self-report.

### 8.2 PSMM Factors (captured in the platform)

- **Task SMM** (4 items): route understanding, landmarks, obstacles, position
- **Team SMM** (4 items): anticipation, role clarity, communication, resolution

### 8.3 Expected Results

- Moderate correlation between KG Jaccard and Task SMM (r ≈ 0.3–0.5)
- KG similarity adds incremental validity over PSMM for predicting IoU (hierarchical regression: PSMM → PSMM + KG Jaccard)
- Team SMM correlates with communication quality metrics but not directly with KG structure

---

## 9. Statistical Analyses

### 9.1 Analysis 1: KG Similarity → Performance

**Model** (mixed-effects):
```
IoU ~ KG_Jaccard_nodes + KG_Jaccard_edges + Route_Edit_Distance + (1|Dyad)
```

Logistic variant for pass/fail.

### 9.2 Analysis 2: KG vs. PSMM as Performance Predictor

**Hierarchical regression**:
- Step 1: `IoU ~ PSMM_task + PSMM_team + (1|Dyad)`
- Step 2: `IoU ~ PSMM_task + PSMM_team + KG_Jaccard + Route_Edit_Distance + (1|Dyad)`

Test: significant ΔR² from Step 1 to Step 2?

### 9.3 Analysis 3: Temporal Convergence

**Growth curve model**:
```
KG_similarity(t) ~ time + time² + outcome(pass/fail) + time × outcome + (1 + time|Dyad)
```

Tests whether convergence rate differs between successful and failing trials.

### 9.4 Analysis 4: Structural Predictors of Failure

Compare KG properties between pass/fail trials:
- Node count, edge count, edge density
- Spatial frame consistency (% allocentric)
- Route landmark recall and precision
- Spatial relation accuracy vs. GT

### 9.5 Analysis 5: Mediation

**Path**:
```
Communication Quality (LLM eval) → KG Alignment (Jaccard) → Performance (IoU)
```

Bootstrapped mediation with `lavaan` or `pingouin`.

---

## 10. Expected Results

### 10.1 Primary (high confidence)

- KG node/edge Jaccard will correlate positively with IoU (r ≈ 0.4–0.6)
- Route sequence edit distance will negatively predict IoU
- Failing trials will have lower landmark recall and more spatial frame switches

### 10.2 Secondary (medium confidence)

- KG similarity increases over the course of successful trials but not failing ones
- Dialogue-derived KG adds 10–20% variance explained beyond PSMM
- REPAIR dialogue acts are associated with sudden KG restructuring (edge deletions/additions)

### 10.3 Novel (the paper's contribution)

- First use of LLM-extracted spatial knowledge graphs to measure SMM structure from dialogue in real time
- Temporal KG convergence as a dynamic indicator of common ground building
- Comparison of dialogue-derived (behavioral) vs. self-reported (PSMM) SMM measurement
- Specific structural features that distinguish success from failure (frame alignment, landmark precision, route sequencing)

---

## 11. Failure Mode Taxonomy (from KG Analysis)

| Failure Mode | KG Signature | Observable in Dialogue |
|--------------|-------------|----------------------|
| **Referential ambiguity** | Multiple landmarks with same relation to route | "The tree... no, the other tree" |
| **Frame misalignment** | High egocentric/allocentric switching | "Go left" (whose left?) |
| **Route order confusion** | High edit distance between Director and Matcher route sequences | Matcher draws segment out of order |
| **Landmark omission** | Key route landmark absent from Matcher's KG | Director mentions landmark Matcher doesn't fixate/reference |
| **Phantom landmarks** | Matcher KG contains landmarks not on their map | Matcher identifies wrong feature as named landmark |
| **Spatial relation error** | Edge in Matcher KG contradicts GT geometry | "The lake is above the church" (it's below) |

---

## 12. What Needs to Be Built

| Component | Priority | Status | Effort |
|-----------|----------|--------|--------|
| **Run pilot sessions (4–6 dyads)** | Critical | Not started | Lab time |
| **Per-speaker graph construction** | High | Partially done | Low — split existing by speaker |
| **GT graph from route JSON** | High | Not built | Medium |
| **Graph comparison metrics** (Jaccard, edit distance, QAP) | High | Not built | Medium |
| **Temporal windowed KG extraction** | High | Not built | Medium |
| **Convergence statistics** (slope, plateau) | Medium | Not built | Low |
| **Hierarchical regression** (KG vs PSMM) | Medium | Not built | Low |
| **Mediation analysis** | Medium | Not built | Low |
| **Failure mode classifier** from KG properties | Medium | Not built | Medium |
| **Run full study (30–40 dyads)** | Critical | Not started | Major lab effort |
| **PSMM → KG correlation analysis** | Medium | Not built | Low |

---

## 13. Key References

- Anderson et al. (1991) — HCRC Map Task Corpus
- Clark & Wilkes-Gibbs (1986) — Referring as a collaborative process (common ground)
- DeChurch & Mesmer-Magnus (2010) — Measuring Shared Mental Models meta-analysis
- van Rensburg et al. (2022) — Five-factor Perceived Shared Mental Models scale
- Mathieu et al. (2000) — Influence of SMMs on team process and performance
- Langan-Fox et al. (2000) — Team mental models: methods, measures, and metrics
- Mohammed et al. (2010) — Metaphor no more: SMM assessment methods
- Stout et al. (1999) — Planning, shared mental models, and coordinated performance
- Richardson & Dale (2005) — Looking to understand (gaze coupling, relevant for multimodal integration)
