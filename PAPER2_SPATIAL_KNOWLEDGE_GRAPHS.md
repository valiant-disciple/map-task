# Paper 2: Spatial Knowledge Graphs from Collaborative Dialogue — Measuring Shared Mental Model Structure in Real Time

## Target Venue
Cognitive Science / Journal of Experimental Psychology: Applied / Frontiers in Psychology

---

## 1. Central Thesis

Shared mental models (SMMs) in collaborative spatial tasks are traditionally measured via post-hoc self-report or offline structural elicitation (card sorts, concept maps, Pathfinder ratings). We propose a new method: extracting **spatial knowledge graphs** directly from naturalistic dialogue using large language models, then comparing graph topology between partners as a behavioral measure of structural SMM alignment. We show that: (a) dialogue-derived KG similarity predicts task performance better than self-reported perceived SMM, (b) KG structure evolves within trials as partners build common ground, and (c) the structural gap between dialogue KG and ground-truth map structure reveals specific failure modes.

---

## 2. Literature Review & Competitive Landscape

### 2.1 Structural SMM Measurement: The Pathfinder Tradition

The standard approach to measuring structural SMMs is the **Pathfinder network** (Schvaneveldt, 1990), where participants rate relatedness of concept pairs and the resulting proximity matrices are compared via QAP correlation or network similarity:

- **Mathieu et al. (JASP, 2000)**: Seminal paper showing Pathfinder-derived team mental model similarity predicts team process and performance. Used explicit concept-pair ratings elicited post-task.

- **Lim & Klein (2006)**: Pathfinder SMM similarity → team adaptation under novel conditions.

- **Smith-Jentsch et al. (2005)**: Pathfinder for team SMM in military command-and-control.

- **Mohammed, Ferzandi & Hamilton (2010)**: "Metaphor no more" — comprehensive review of SMM measurement methods (Pathfinder, concept mapping, card sorting, causal mapping). All require **explicit elicitation** — participants must provide ratings or sort cards.

- **DeChurch & Mesmer-Magnus (Group Dynamics, 2010)**: Meta-analysis of SMM measurement. Found that structural measures (Pathfinder) show stronger relationships with performance than perceived measures (surveys). Key implication: structure matters more than perception.

**Critical limitation of all Pathfinder-based work**: Requires explicit, time-consuming elicitation. Cannot capture dynamics (how SMMs evolve during the task). Cannot scale to real-time measurement. Reflects what participants *report* they know, not what they *actually communicated*.

### 2.2 Perceived SMM Measurement (Survey-Based)

- **van Rensburg et al. (Frontiers, 2022)**: Five-factor Perceived SMM scale (Equipment, Execution, Interaction, Composition, Temporal). Our PSMM survey is based on this. Quick to administer but captures perception, not actual knowledge structure.

- **Bello, Aqlan & Salas (Small Group Research, 2025)**: Recent review of shared mental models in virtual teams — concludes that better measurement methods are needed, especially for capturing dynamic evolution of SMMs.

**Gap**: Surveys capture *perceived* alignment. Pathfinder captures *structural* alignment but requires explicit elicitation. Neither captures what partners actually communicated and understood *during* the task.

### 2.3 NLP and Dialogue Analysis for Team Cognition

- **Lexical alignment / LSA approaches**: Fusaroli et al. (2012) used CRQA on linguistic tokens to measure dialogue convergence. Duran & Dale (2014) applied recurrence analysis to conversation. These measure *lexical* alignment (do partners use the same words?), not *structural* alignment (do partners have the same spatial knowledge?).

- **Garrod & Doherty (1994)**: Conceptual pacts in map task — partners converge on referring expressions. Measured lexically, not structurally.

- **Brennan & Clark (1996)**: Conceptual pacts and reference — again, lexical convergence, not graph structure.

- **Tenbrink et al. (2011, 2024)**: Fine-grained linguistic analysis of spatial language (perspectives, frames of reference). Produces linguistic annotations but not structured graphs.

**Gap**: The dialogue analysis literature measures *linguistic* convergence (words, syntax, prosody). Nobody measures *structural-conceptual* convergence (do partners build the same graph of spatial relationships?).

### 2.4 LLMs for Knowledge Graph Construction

The use of LLMs for KG construction from text is a rapidly growing field:

- **Zhu et al. (arXiv, 2023)**: Survey of LLMs for KG construction and reasoning. GPT-4 excels at relation extraction and entity recognition from unstructured text. KGGen (2025) further refines this.

- **Nature Communications (2025)**: LLM-powered KG construction for mental health exploration — shows that the LLM→KG pipeline is accepted in high-impact journals for domain-specific applications.

- **SpatialRGPT (2024)**: Grounded spatial reasoning in vision-language models — constructs 3D scene graphs from images. Relevant for our vision-based landmark extraction.

- **Dialogue-specific KG**: Xu et al. (2023–2024) built KGs from dialogue for chatbot knowledge grounding. But for dialogue *systems*, not cognitive measurement.

**Gap**: LLMs are used to extract KGs from text (general NLP) and from images (computer vision). Nobody has used LLMs to extract **spatial knowledge graphs from collaborative task dialogue** as a measure of **shared mental models**. The cognitive science application is completely unexplored.

### 2.5 Common Ground and Temporal Convergence

- **Clark (1996)**: Grounding theory — common ground accumulates through grounding acts (assertions, acknowledgments, repairs). The process is inherently temporal.

- **Clark & Wilkes-Gibbs (1986)**: Referring expressions become shorter as common ground builds — the "collaborative process" of reference.

- **Fusaroli et al. (2012)**: CRQA on dialogue tokens shows temporal structure of linguistic alignment. But measures words, not concepts.

- **Mills (2014)**: Common ground and spatial perspective-taking — theoretical, not structural measurement.

**Gap**: Common ground is theorized to accumulate temporally, but nobody has measured this accumulation as *graph convergence*. Tracking how two people's KGs become more similar over the course of a conversation is a new operationalization of grounding.

### 2.6 The Map Task Paradigm

- **Anderson et al. (1991)**: HCRC Map Task Corpus — extensively studied for dialogue structure, but never for structural knowledge representation.

- **Pickering & Garrod (2004, 2021)**: Interactive alignment in dialogue — partners converge on representations. Measured via lexical and syntactic priming, not graph structure.

- No published map task study has extracted knowledge graphs from the dialogue or compared structural representations between participants.

### 2.7 What Is Genuinely Novel

1. **LLM-extracted spatial KGs from task dialogue**: Nobody has done this. The method bridges NLP (LLM KG extraction), spatial cognition (mental model theory), and team science (SMM measurement).

2. **Automated structural SMM from behavior**: Unlike Pathfinder (explicit elicitation) or PSMM (self-report), our method derives structure from what participants *actually said*. No elicitation burden. Ecological validity.

3. **Temporal KG convergence**: Tracking graph similarity over time within a trial as a dynamic measure of common ground building. New operationalization of grounding theory.

4. **Dialogue-derived vs. self-report SMM comparison**: Direct empirical test of whether behavioral structural measurement (KG) outperforms perceived measurement (PSMM).

5. **Failure mode taxonomy from KG structure**: Specific graph signatures for referential ambiguity, frame misalignment, route confusion — actionable diagnostics.

**Honest assessment**: This is genuinely novel — the specific intersection of LLM KG extraction + spatial dialogue + SMM measurement does not exist in the literature as of March 2026. The risk is not novelty but **reliability**: can GPT-4 extract accurate spatial KGs from disfluent, overlapping, real-world dialogue? Reviewers will demand validation against human coders. If the extraction works, this paper has a strong novelty claim for Cognitive Science or JEP: Applied.

---

## 3. Research Questions

**RQ1**: Can spatial knowledge graphs extracted from collaborative dialogue via LLMs capture meaningful structural differences between dyads?

**RQ2**: Does dialogue-derived KG similarity predict task performance (IoU, pass/fail) beyond self-reported PSMM?

**RQ3**: How does KG structure evolve within a trial as partners build common ground?

**RQ4**: What structural properties of dialogue-derived KGs distinguish successful from unsuccessful trials?

**RQ5**: Does KG alignment mediate the relationship between communication quality and task performance?

---

## 4. Hypotheses

**H1**: Dyads with higher KG similarity (node Jaccard, edge Jaccard) achieve higher drawing accuracy (IoU) and faster completion.
- *Basis*: DeChurch & Mesmer-Magnus 2010 (structural SMM → performance, ρ = 0.39 in meta-analysis).

**H2**: Dialogue-derived KG similarity explains additional variance in performance beyond PSMM self-report.
- *Basis*: DeChurch & Mesmer-Magnus 2010 found structural measures outperform perceived measures. Our KG is a behavioral structural measure.

**H3**: KG similarity increases over the course of successful trials (convergence) but not failing trials. Rate of convergence predicts final accuracy.
- *Basis*: Clark & Wilkes-Gibbs 1986 (collaborative convergence); Fusaroli et al. 2012 (temporal alignment dynamics).

**H4**: Failing trials show: (a) lower landmark agreement, (b) more spatial frame switches, (c) more REPAIR acts, (d) greater divergence from ground truth.
- *Basis*: Tenbrink (frame of reference confusion → errors); map task repair literature.

**H5**: Communication quality → KG alignment → Performance, with KG alignment mediating.
- *Basis*: SMM → process → performance mediation framework (Mathieu et al. 2000).

---

## 5. Experimental Design

### 5.1 Task
Same as Paper 1: Director–Matcher Map Task, 8 trials (2 warmup + 6 data), 210 s each.

### 5.2 Participants

**Target**: 30 dyads (60 participants), same pool as Paper 1.

KG metrics are noisier than physiological measures (LLM extraction has variance). With 6 trials × 30 dyads = 180 observations, adequate for mixed-effects models with 4–5 fixed effects. 25 dyads is the minimum.

### 5.3 Data Required Per Trial

| Input | Source | Purpose |
|-------|--------|---------|
| ASR transcript (Director) | Whisper word-level | KG extraction, dialogue acts |
| ASR transcript (Matcher) | Whisper word-level | KG extraction, dialogue acts |
| Map images | map{N}f.gif, map{N}g.gif | Landmark extraction (vision) |
| Ground truth route | gt_{N}.json | Route accuracy validation |
| Drawing strokes | Canvas events | IoU computation (outcome) |
| PSMM responses | Post-trial survey | Comparison with KG similarity |
| Dialogue act classification | GPT-4.1-mini | REPAIR count, communication quality |

---

## 6. Knowledge Graph Construction

### 6.1 Step 1: Landmark Extraction (Vision)

GPT-4.1 vision API processes Director and Matcher map images. Per landmark: name, type, position (x%, y%), size, route proximity.

**Status**: Implemented in `knowledge_graph.py` with caching.

### 6.2 Step 2: Spatial Relation Extraction (Text)

GPT-4.1 text API processes per-speaker ASR transcripts. Extracts:
- Spatial relations: source landmark → target landmark, relation type (NEAR, LEFT_OF, RIGHT_OF, ABOVE, BELOW, BETWEEN, ALONG, PAST, AROUND, THROUGH), confidence, speaker
- Route sequence: ordered landmark list
- Spatial frame: egocentric / allocentric / mixed
- Frame switches count

**Status**: Implemented in `knowledge_graph.py`.

### 6.3 Step 3: Per-Speaker Graph Construction

Separate directed graphs for Director and Matcher:
- **Nodes** = landmarks mentioned by that speaker
- **Edges** = spatial relations uttered by that speaker
- Edge attributes: relation_type, confidence, count (repeated mentions)

**Status**: Partially implemented (currently builds combined graph). **Needs**: split by speaker field.

### 6.4 Step 4: Ground Truth Graph

From `gt_{N}.json` route coordinates + vision-extracted landmarks:
- Nodes = all landmarks within proximity threshold of route
- Edges = pairwise spatial relations computed from (x, y) positions
- Route sequence from GT coordinate order

**Status**: **Not built.**

### 6.5 Validation: LLM Extraction vs. Human Coders

**Critical for publication.** Sample 20 trial transcripts, have 2 human coders independently extract:
- Landmarks mentioned
- Spatial relations between landmarks
- Route sequence

Compare LLM extraction to human coding:
- Node agreement (Cohen's κ for landmark identification)
- Edge agreement (κ for spatial relations)
- Route order correlation (Kendall τ)

Target: κ ≥ 0.70 (substantial agreement) for each metric.

**Status**: **Not built.** Needs coding scheme + human coders.

---

## 7. Graph Comparison Metrics

### 7.1 Director–Matcher Similarity (SMM Alignment)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Node Jaccard | \|V_D ∩ V_M\| / \|V_D ∪ V_M\| | Shared landmark vocabulary |
| Edge Jaccard | \|E_D ∩ E_M\| / \|E_D ∪ E_M\| | Shared relational structure |
| Typed Edge Jaccard | Edge match requires same relation type | Strict structural agreement |
| Route Edit Distance | Levenshtein(route_D, route_M) | Sequential alignment |
| Route Kendall τ | Rank correlation of shared landmarks in route | Monotonic route agreement |
| Frame Agreement | % trials both use same spatial frame | Reference frame alignment |
| QAP Correlation | Correlation of adjacency matrices (permutation test) | Classic SMM similarity (Pathfinder tradition) |

### 7.2 KG vs. Ground Truth (Accuracy)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Landmark Recall | \|mentioned ∩ GT_route\| / \|GT_route\| | Coverage of critical landmarks |
| Landmark Precision | \|mentioned ∩ GT_route\| / \|mentioned\| | Focus on relevant landmarks |
| Route Order Accuracy | 1 − normalized Levenshtein(mentioned_route, GT_route) | Sequential correctness |
| Relation Accuracy | % relations consistent with GT geometry | Factual spatial correctness |

**Status for all metrics**: **Not built.** Low-medium effort each.

---

## 8. Temporal Evolution of Knowledge Graphs

### 8.1 Method

Divide each trial transcript into temporal windows (60 s, 30 s overlap) and extract cumulative KGs at each window boundary. Compute Director–Matcher similarity at each step.

### 8.2 Metrics Over Time

- Landmark accumulation curve: new landmarks per window
- Edge density: edges / possible edges
- D–M Node Jaccard(t): convergence trajectory
- Convergence rate: slope of linear fit to Jaccard(t)
- Convergence plateau: time at which Jaccard first exceeds 80% of final value

### 8.3 Expected Pattern

Successful trials: monotonically increasing KG similarity → plateau before trial end.
Failing trials: flat or non-monotonic trajectory → no convergence.

**Status**: **Not built.** Requires windowed transcript splitting + per-window GPT extraction + cumulative graph union.

---

## 9. Dialogue Act Integration

### 9.1 Dialogue Acts (from llm_eval.py — implemented)

10 categories: INSTRUCT, DESCRIBE, CHECK, QUERY, CLARIFY, ACKNOWLEDGE, REPAIR, META, FILLER, OTHER

### 9.2 Communication Quality (from llm_eval.py — implemented)

8 dimensions (1–7): clarity, specificity, efficiency, grounding, adaptiveness, spatial_precision, collaboration, overall

### 9.3 Predictions

| Dialogue Feature | KG Prediction |
|-----------------|---------------|
| More INSTRUCT + DESCRIBE | Richer KGs (more nodes, edges) |
| More REPAIR acts | Lower KG similarity (misalignment being corrected) |
| Higher CHECK/ACKNOWLEDGE rate | Faster convergence (grounding loop) |
| Higher spatial_precision score | More accurate spatial relations in KG |
| Higher grounding score | Higher D–M Jaccard |

---

## 10. Comparison: KG vs. PSMM Self-Report

### 10.1 The Validation Question

Does dialogue-derived KG similarity correlate with PSMM self-report? If yes → convergent validity. If KG predicts performance *beyond* PSMM → incremental validity (behavioral > self-report).

### 10.2 PSMM Factors (captured per trial)

- Task SMM (4 items): route understanding, landmarks, obstacles, position
- Team SMM (4 items): anticipation, role clarity, communication, resolution

### 10.3 Expected Results

- Moderate correlation: KG Jaccard × Task SMM (r ≈ 0.3–0.5)
- Incremental validity: KG adds 10–20% ΔR² beyond PSMM in predicting IoU
- Team SMM correlates with communication quality but not KG structure directly

---

## 11. Statistical Analyses

### 11.1 KG Similarity → Performance (Mixed-Effects)
```
IoU ~ KG_Node_Jaccard + KG_Edge_Jaccard + Route_Edit_Distance + (1|Dyad)
```

### 11.2 KG vs. PSMM (Hierarchical)
- Step 1: `IoU ~ PSMM_task + PSMM_team + (1|Dyad)`
- Step 2: `IoU ~ PSMM_task + PSMM_team + KG_Jaccard + Route_Edit_Distance + (1|Dyad)`
- Test: ΔR², likelihood ratio

### 11.3 Temporal Convergence (Growth Curve)
```
KG_similarity(t) ~ time + time² + outcome + time×outcome + (1 + time|Dyad)
```

### 11.4 Structural Predictors of Failure (Logistic Mixed-Effects)
```
Pass/Fail ~ node_count + edge_density + frame_consistency + landmark_recall + route_accuracy + (1|Dyad)
```

### 11.5 Mediation
```
Communication Quality → KG Alignment → Performance (IoU)
```
Bootstrapped (5000 resamples), `lavaan` or `pingouin`.

**Status**: All analyses **not built.**

---

## 12. Expected Results

### 12.1 High Confidence

- KG node/edge Jaccard correlates positively with IoU (r ≈ 0.4–0.6). *Basis*: DeChurch & Mesmer-Magnus 2010 meta-analysis found ρ = 0.39 for structural SMM → performance.
- Failing trials have lower landmark recall and more frame switches. *Basis*: well-documented in spatial communication literature.

### 12.2 Medium Confidence

- KG similarity increases during successful trials, not failing ones. *Basis*: grounding theory (Clark 1996) predicts convergence.
- KG adds 10–20% variance beyond PSMM. *Basis*: structural > perceived in meta-analysis.
- REPAIR acts associated with KG restructuring (edge changes).

### 12.3 Novel Claims

- LLM-extracted spatial KGs are a valid and reliable measure of structural SMM (κ ≥ 0.70 vs. human coders)
- Temporal KG convergence rate predicts final task accuracy
- Specific failure mode taxonomy identifiable from graph properties
- Behavioral structural SMM (KG) outperforms self-report (PSMM) — the methodological advance

---

## 13. Failure Mode Taxonomy

| Failure Mode | KG Signature | Dialogue Marker |
|--------------|-------------|-----------------|
| **Referential ambiguity** | Multiple landmarks with overlapping relations | "The tree... no, the other tree" |
| **Frame misalignment** | High ego/allo switching; contradictory LEFT_OF edges | "Go left" — whose left? |
| **Route order confusion** | High edit distance D vs. M route sequences | Matcher draws segments out of order |
| **Landmark omission** | Route landmark absent from Matcher KG | Director mentions it, Matcher never references it |
| **Phantom landmarks** | Matcher KG has nodes absent from both maps | Matcher misidentifies a feature |
| **Spatial relation error** | Edge in Matcher KG contradicts GT geometry | "Lake is above the church" (it's below) |

---

## 14. Risks and Limitations

| Risk | Mitigation |
|------|-----------|
| LLM extraction unreliable for disfluent speech | Validate against human coders (target κ ≥ 0.70); report extraction quality metrics |
| ASR errors corrupt landmark names | Use fuzzy matching for landmark identification; report WER |
| Short trials (210s) may produce sparse KGs | 210s of active dialogue typically contains 50–150 utterances — sufficient for 5–15 landmark mentions |
| GPT-4 extraction is not deterministic | Run extraction 3× per trial, use majority vote for edges; report inter-run agreement |
| Temporal windowed extraction is API-expensive | Budget: ~$0.50/trial × 4 windows × 180 trials = ~$360 total |
| QAP permutation test is computationally expensive for large graphs | Map task KGs are small (5–15 nodes) — QAP is fast |
| Reviewers may question LLM as a "black box" | Report prompts, extraction schemas, and example outputs in supplementary materials |

---

## 15. What Needs to Be Built

| Component | Priority | Status | Effort |
|-----------|----------|--------|--------|
| **Run pilot (4–6 dyads)** | Critical | Not started | Lab time |
| **Per-speaker graph construction** | High | Partial | Low |
| **GT graph from route JSON** | High | Not built | Medium |
| **Graph comparison metrics** (Jaccard, edit distance, QAP) | High | Not built | Medium |
| **Temporal windowed KG extraction** | High | Not built | Medium |
| **LLM extraction validation** (coding scheme + human coders) | High | Not built | High |
| **Convergence statistics** (slope, plateau) | Medium | Not built | Low |
| **Hierarchical regression** (KG vs PSMM) | Medium | Not built | Low |
| **Growth curve model** (temporal convergence) | Medium | Not built | Low |
| **Mediation analysis** | Medium | Not built | Low |
| **Run full study (30 dyads)** | Critical | Not started | Major |

---

## 16. References

### Shared Mental Models — Structural Measurement
- Schvaneveldt, R. W. (1990). *Pathfinder associative networks: Studies in knowledge organization*. Ablex.
- Mathieu, J. E., et al. (2000). The influence of shared mental models on team process and performance. *JASP*, 85(2), 273–283.
- DeChurch, L. A. & Mesmer-Magnus, J. R. (2010). Measuring shared mental models: Meta-analysis. *Group Dynamics*, 14(1), 1–14.
- Mohammed, S., Ferzandi, L. & Hamilton, K. (2010). Metaphor no more: A 15-year review of the team mental model construct. *JOMM*, 16(4), 461–480.
- Lim, B. C. & Klein, K. J. (2006). Team mental models and team performance. *JASP*, 91(6), 1244–1253.
- Smith-Jentsch, K. A., et al. (2005). Investigating linear and interactive effects of SMM on safety and efficiency. *JASP*, 90(3), 523–535.

### Shared Mental Models — Perceived Measurement
- van Rensburg, J. J., et al. (2022). Five-factor Perceived SMM scale. *Frontiers in Psychology*, 12, 784200.
- Bello, K., Aqlan, F. & Salas, E. (2025). Understanding shared mental models in virtual teams. *Small Group Research*. https://journals.sagepub.com/doi/10.1177/10464964251395005

### Common Ground & Dialogue
- Clark, H. H. (1996). *Using language*. Cambridge University Press.
- Clark, H. H. & Wilkes-Gibbs, D. (1986). Referring as a collaborative process. *Cognition*, 22(1), 1–39.
- Brennan, S. E. & Clark, H. H. (1996). Conceptual pacts and lexical choice in conversation. *JEP:LMC*, 22(6), 1482–1493.
- Garrod, S. & Doherty, G. (1994). Conversation, co-ordination and convention. *Cognition*, 53(3), 181–215.
- Pickering, M. J. & Garrod, S. (2004). Toward a mechanistic psychology of dialogue. *BBS*, 27(2), 169–226.
- Fusaroli, R., et al. (2012). Coming to terms: Quantifying the benefits of linguistic coordination. *Psychological Science*, 23(8), 931–939.
- Duran, N. D. & Dale, R. (2014). Perspective-taking in dialogue as self-organization under social constraints. *New Ideas in Psychology*, 32, 131–146.

### Spatial Cognition & Language
- Tversky, B. (1993). Cognitive maps, cognitive collages, and spatial mental models. *COSIT*, 14–24.
- Taylor, H. A. & Tversky, B. (1992). Spatial mental models derived from survey and route descriptions. *JML*, 31(2), 261–292.
- Tenbrink, T. (2011). Reference frames of space and time in language. *Journal of Pragmatics*, 43(3), 704–722.

### LLMs for Knowledge Graph Construction
- Zhu, Y., et al. (2023). LLMs for knowledge graph construction and reasoning: Recent capabilities and future opportunities. *arXiv:2305.13168*. https://arxiv.org/abs/2305.13168
- Nature Communications (2025). LLM-powered knowledge graph construction for mental health exploration. https://www.nature.com/articles/s41467-025-62781-z
- KGGen (2025). Extracting knowledge graphs from plain text with language models. https://arxiv.org/html/2502.09956v1

### Map Task
- Anderson, A. H., et al. (1991). The HCRC Map Task Corpus. *Language and Speech*, 34(4), 351–366.

### Eye Tracking (Cross-Paper Reference)
- Richardson, D. C. & Dale, R. (2005). Looking to understand. *Cognitive Science*. https://doi.org/10.1207/s15516709cog0000_29
