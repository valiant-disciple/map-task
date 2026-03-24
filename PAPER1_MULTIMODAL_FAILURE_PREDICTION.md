# Paper 1: A Real-Time Multimodal Team Workload Index Predicts Coordination Failures in Dyadic Spatial Collaboration

## Target Venue
PNAS / Psychophysiology / Human Factors (Nature Human Behaviour if results are very strong)

---

## 1. Central Thesis

Coordination failures in dyadic spatial tasks are preceded by detectable multimodal signatures — drops in physiological synchrony, gaze desynchronization, speech disfluency, and rising workload — that emerge 10–60 seconds before overt errors. We introduce W_team(t), a live Team Workload Index integrating heart rate, gaze, speech, and subjective load, and demonstrate that it predicts trial-level pass/fail and moment-to-moment accuracy better than any single modality alone. The method generalizes to high-stakes operator settings (nuclear/chemical control rooms).

---

## 2. Literature Review & Competitive Landscape

### 2.1 Physiological Synchrony Predicts Team Performance

The link between interpersonal physiological synchrony (IPS) and team outcomes is now well-established:

- **Tognoli et al. (PNAS, 2024)**: HR synchrony predicted whether 44 groups (n=204) reached correct consensus in a Hidden Profile decision task with >70% cross-validation accuracy — significantly higher than discussion duration, subjective team function ratings, or baseline HR alone. This is the strongest evidence to date that HR coupling is a biomarker of effective information processing. However, the study used HR as the sole physiological measure — no gaze, speech, or composite index.

- **Physiological synchrony + behavioral synchrony (Nature Scientific Reports, 2020)**: Physiological and behavioral synchrony predicted group cohesion and performance, but again with single modalities, not composites.

- **Collaborative learning quality (PMC, 2021)**: EDA and HR synchrony during classroom group discussions distinguished high- vs. low-collaboration dyads. Limited to two physiological channels.

- **Dyadic joint action (Psychophysiology, 2025)**: 78 dyads showed increased IPS during novel tasks, with social anxiety reducing synchrony. Largest dyadic IPS sample to date. Used HR/HRV only.

- **Nature Reviews Psychology (2026)**: Comprehensive meta-review of IPS identified significant heterogeneity in methods and called for multimodal approaches as a key future direction.

**Gap**: All these studies use 1–2 physiological channels. None combine HR + gaze + speech + behavioral measures into a unified team-level index.

### 2.2 RQA/CRQA/MdRQA for Team Coordination

- **Wallot & Leonardi (Frontiers, 2018)**: Definitive tutorial for CRQA, DCRP, and MdRQA applied to behavioral/physiological time series. Our pipeline follows this methodology.

- **Wallot, Roepstorff & Monseter (Frontiers, 2016)**: Original MdRQA paper. Joint phase space captures system-level dynamics invisible to pairwise analysis.

- **Physiological synchrony + task performance (MDPI Sensors, 2023)**: MdRQA on 3-member teams (HR + EDA) predicted task performance and frustration. Team-level, but only 2 physiological channels, no gaze or speech.

- **van Eijndhoven et al. (Group Dynamics, 2025)**: **Closest competitor.** Used windowed synchronization coefficient (SC) and windowed MdRQA on PPG + EDA to detect team coordination breakdowns in crisis scenarios. They manually identified breakdown events and compared physiological features in breakdown vs. non-breakdown windows. Found that cardiovascular and skin conductance signals showed the strongest correspondence with coordination transitions. However: only PPG + EDA (no gaze, no speech, no composite index), and detection was retrospective, not predictive.

**Gap**: van Eijndhoven 2025 detected breakdowns retrospectively from physiology. We propose to **predict** them in advance using a multimodal composite, and add gaze, speech, and drawing behavior to the physiological channels.

### 2.3 Multimodal Cognitive Workload

- **Lucchese et al. (IET, 2025)**: Systematic review of cognitive workload methods — identifies multimodal fusion as the trend but notes that existing indices are individual-level, not team-level.

- **IEEE (2023)**: Multimodal physiological cognitive load prediction achieved 81% accuracy using HR + pupil + EEG. Individual-level.

- **Frontiers Human Neuroscience (2021)**: Pilot-UAV teaming workload estimation from physiological features. Individual-level, not dyadic.

- **PMC (2023)**: Multimodal assessment in smart factory settings — EEG + eye tracking + subjective. Individual-level.

**Gap**: Multimodal workload indices exist for **individuals**. No published work proposes a **team-level** real-time workload index that integrates synchrony, inequity, and joint attention alongside individual physiological load.

### 2.4 The Map Task Paradigm

- **Anderson et al. (1991)**: Original HCRC Map Task Corpus. Extensively studied for dialogue, turn-taking, referring expressions, and prosody. The gold standard referent task for our work.

- The map task literature is overwhelmingly **speech-only**. No published map task study combines HR + eye tracking + drawing behavior. Our platform is the first to provide synchronized multimodal capture for the map task.

### 2.5 Surrogate Testing

- **Tschacher & Meier (Entropy, 2021)**: mv-SUSY framework for multivariate surrogate synchrony. Any IPS result must be tested against pseudo-dyad baselines. Our pipeline includes 20 time-shift surrogates with z-score significance testing — this is implemented in postprocess.py.

### 2.6 What Is Genuinely Novel

1. **W_team(t)**: First real-time **team-level** workload index combining individual physiological load, workload inequity (Gini), HR synchrony (%DET), and joint gaze attention.
2. **Multimodal map task**: First map task study with synchronized HR + eye tracking + speech + drawing behavior + subjective workload.
3. **Predictive pre-failure signatures**: Multimodal temporal signatures preceding failures (not just retrospective detection like van Eijndhoven).
4. **Control-room translation**: Direct analytical bridge from map task to operator micro-world.

**Honest assessment**: The novelty is in the **integration**, not the individual components. Each piece (HR synchrony → performance, gaze entropy → workload, RQA methods) is established. The contribution is showing that the whole exceeds the sum of parts. For PNAS/Psychophysiology this is a strong paper. For Nature Human Behaviour it's borderline — would need a very striking result (e.g., W_team predicts failures 30+ seconds in advance with AUC > 0.85).

---

## 3. Research Questions

**RQ1**: Can coordination failures be predicted from multimodal physiological and behavioral signals before they manifest as task errors?

**RQ2**: Does W_team(t) outperform single-modality predictors (HR alone, gaze alone, TLX alone) of task performance?

**RQ3**: Are pre-failure windows (−60 s to 0 s before error onset) characterized by distinct multimodal signatures compared to successful segments?

**RQ4**: Does physiological synchrony mediate the relationship between shared mental models and task performance?

---

## 4. Hypotheses

**H1**: Higher HR synchrony (%DET_HR) and gaze synchrony (%DET_gaze) predict better task accuracy (higher IoU) and fewer repair episodes.
- *Basis*: PNAS 2024 (HR synchrony → group performance), Psychophysiology 2025 (IPS → joint action success).

**H2**: Pre-failure windows (−60 s to 0 s) show:
- (a) HR %DET drops 10–30 s before large errors
- (b) Gaze entropy spikes (less efficient visual search)
- (c) Speech overlap increases and REPAIR dialogue acts increase
- (d) W_team(t) rises preceding the failure
- *Basis*: van Eijndhoven 2025 (physiological features distinguish coordination breakdowns), Nature Reviews Psych 2026 (desynchronization precedes performance declines).

**H3**: W_team(t) explains significantly more variance in performance than NASA-TLX alone or any single physiological channel.
- *Basis*: No direct precedent — this is the core novel claim.

**H4**: Structural SMM similarity → process synchrony (HR %DET + gaze joint attention) → performance, with synchrony partially mediating.
- *Basis*: DeChurch & Mesmer-Magnus 2010 (SMM → performance), Mathieu et al. 2000 (SMM → process → performance mediation).

---

## 5. Experimental Design

### 5.1 Task
Director–Matcher Map Task. 8 trials (2 warmup + 6 data), 210 s each, role swap mid-session. Maps have subtly different landmarks between roles. Difficulty modulated by map complexity and landmark ambiguity.

### 5.2 Participants

**Target**: 30 dyads (60 participants). Justification:

| Study | N (groups/dyads) | N (individuals) | Design |
|-------|-------------------|-----------------|--------|
| PNAS 2024 HR synchrony | 44 groups | 204 | 1 task per group |
| van Eijndhoven 2025 | 21 teams | 63 | Multiple sessions |
| Psychophysiology 2025 (dyadic) | 78 dyads | 156 | Multiple conditions |
| Sensors 2023 (3-member MdRQA) | 16 teams | 48 | 1 task per team |
| **Our study** | **30 dyads** | **60** | **6 data trials per dyad = 180 observations** |

Mixed-effects models require ≥ 20 level-2 units (dyads) for stable random-effect estimates. With 6 trials per dyad = 180 trial-level observations for 4–6 fixed effects: adequate power for medium effects (d = 0.5, power > 0.80). 30 dyads gives safety margin.

**Minimum viable**: 20 dyads (40 participants, 120 observations). Below 20 = underpowered.

**Pilot**: 4–6 dyads to calibrate thresholds, map difficulty, and W_team weights.

### 5.3 Data Captured Per Trial

| Modality | Source | Rate | Key Features |
|----------|--------|------|--------------|
| Heart rate | Galaxy Watch 4 → WS → backend | ~1 Hz (BPM) | HRQ, RMSSD, SDNN, ln(RMSSD), pNN50, SampEn, DFA α1, SD1/SD2, HF power |
| Eye tracking | SmartEye Pro 10 or Aurora (iMotions) | 60 Hz | Fixation rate/duration, saccade amplitude, pupil diameter, scanpath entropy, AOI dwell, joint attention |
| Speech | Laptop mic → WebM/Opus → Whisper ASR | Continuous | F0, RMS, speech rate, pauses, turn-taking, dialogue acts (10 categories), repair count |
| Drawing behavior | Canvas strokes (651×900 px) | Continuous | Stroke kinematics, hesitations, erase ratio, coverage growth, fractal dimension |
| Subjective workload | NASA-TLX | Post-trial | 6 dimensions (0–100): mental, physical, temporal, performance, effort, frustration |
| Perceived SMM | PSMM survey | Post-trial | 8 items (1–7 Likert): task SMM (4) + team SMM (4) |
| Trial success | Director rating | Post-trial | Target reached (yes/no), path confidence (1–7) |

### 5.4 Temporal Synchronization

All modalities aligned to Director's reference clock via NTP-style offset measurement (< 20 ms precision) with per-trial SyncFlash recalibration for eye trackers (< 37 ms worst case). Full technical specification in TECHNICAL_METHODS.md.

---

## 6. Outcome Variables

### 6.1 Trial-Level Binary (Pass/Fail)

Pass = ALL of:
- Completed within 210 s
- Drawing accuracy IoU ≥ threshold (pilot-calibrated, target ~70–80% pass rate)
- Director rates target_reached = "yes"

Fail = timeout OR accuracy below threshold OR Director rates failure.

**Status**: IoU, F1, SSIM, Hausdorff, Chamfer, boundary F1 computed in postprocess.py. Director's trial_success form captures target_reached. **Needs**: `label_pass_fail()` function with pilot-calibrated thresholds.

### 6.2 Trial-Level Continuous

- IoU / boundary F1 (drawing accuracy)
- Time to complete
- Number of REPAIR dialogue acts
- NASA-TLX composite (mean of 6 dimensions)

### 6.3 Time-Series (1 Hz)

- W_i(t): per-person workload index
- W_team(t): team workload index
- Time-series drawing accuracy (IoU at every 5th stroke)

---

## 7. The Live Workload Index (Core Contribution)

### 7.1 Individual Workload W_i(t)

```
W_i(t) = α₁·z(HRQ_i) + α₂·z(−RMSSD_i) + α₃·z(GazeEnt_i) + α₄·z(SpeechStress_i) + α₅·z(TLX_i,block)
```

| Component | Definition | Direction | Basis |
|-----------|-----------|-----------|-------|
| HRQ | (HR − HR_baseline) / HR_baseline | Higher = more load | Standard cardiac workload index |
| −RMSSD | Negated RMSSD (ms) | Lower RMSSD = higher sympathetic load | Gu et al. 2023 (validated for ultra-short); PMC 2024 standardization |
| GazeEnt | Scanpath entropy over spatial grid | Higher = less efficient search | Shiferaw et al. 2019; Krejtz et al. 2014 |
| SpeechStress | Composite: z(F0_slope) + z(intensity) + z(speech_rate) | Higher = more stressed speech | Prosodic stress literature |
| TLX | Block-level NASA-TLX composite | Higher = more subjective load | Hart & Staveland 1988 |

- `z(·)`: within-person z-score standardization (relative to that person's baseline)
- `α_k`: weights initialized equal (1/5), calibrated in pilot via ridge regression against trial IoU with leave-one-dyad-out cross-validation
- Smoothing: EWMA with 5 s span, updated at 1 Hz

### 7.2 Team Workload W_team(t)

```
W_team(t) = β₁·W̄_i(t) + β₂·Gini(W_i) + β₃·z(−%DET_HR) + β₄·z(−JointAttn)
```

| Component | Definition | Direction | Basis |
|-----------|-----------|-----------|-------|
| W̄_i | Mean of Director + Matcher W_i(t) | Higher = more team load | Standard aggregation |
| Gini(W_i) | Workload inequity (0=equal, 1=one-sided) | Higher = worse distribution | Novel — captures asymmetric load |
| −%DET_HR | Negated HR synchrony determinism (windowed CRQA) | Lower %DET = desynchronized | PNAS 2024; van Eijndhoven 2025 |
| −JointAttn | Negated joint gaze attention proportion | Lower = less shared attention | Richardson & Dale 2005; Schneider & Pea 2013 |

- `β_k`: initialized equal (1/4), calibrated via regression to predict pass/fail

### 7.3 What Exists vs. What Needs Building

| Component | Status |
|-----------|--------|
| All input features (HR, HRV, gaze entropy, prosody, TLX, CRQA, joint attention) | Computed in postprocess.py |
| Windowed CRQA (60s window, 30s step) | Computed |
| Surrogate baseline testing (20 surrogates) | Computed |
| **`workload.py`** — W_i(t) and W_team(t) computation | **Not built** |
| **EWMA smoothing** at 1 Hz | **Not built** |
| **Gini coefficient** for workload inequity | **Not built** |
| **Weight calibration** (ridge regression with LOO-dyad CV) | **Not built** |

---

## 8. Analyses

### 8.1 Analysis 1: Trial-Level Prediction (Mixed-Effects)

```
IoU ~ Structural_SMM + %DET_HR + NASA_TLX + Repairs + W_team_mean + (1|Dyad)
```

Logistic variant for pass/fail. Compare: full model vs. TLX-only vs. %DET-only vs. W_team-only.

**Method**: `lme4` (R) or `statsmodels` (Python). Report AIC, BIC, marginal/conditional R², AUC-ROC for logistic. **Not built.**

### 8.2 Analysis 2: Event-Aligned Pre-Failure Windows

1. Identify failure events: first stroke where cumulative IoU drops below threshold, or first REPAIR dialogue act in failing trials
2. Extract −60 s to 0 s window before each failure event
3. Extract matched −60 s windows from successful trials (propensity-matched on elapsed time)
4. Compare W_team, %DET_HR, gaze entropy, repair rate trajectories

**Statistical test**: Cluster-based permutation test on trajectory difference (Maris & Oostenveld 2007), or growth curve model with outcome×time interaction.

**Needs**: `event_aligned.py` — window extraction, trajectory comparison code.

### 8.3 Analysis 3: Mediation

```
Structural SMM (PSMM similarity) → Process synchrony (%DET_HR + joint attention) → Performance (IoU)
```

Bootstrapped mediation (5000 resamples) via `lavaan` (R) or `pingouin` (Python). **Not built.**

### 8.4 Analysis 4: W_team Validation (Nested Model Comparison)

| Model | Predictors | Expected AUC |
|-------|-----------|--------------|
| Null | Intercept only | 0.50 |
| TLX only | NASA-TLX composite | 0.60–0.65 |
| HR only | %DET_HR + RMSSD | 0.65–0.70 |
| Gaze only | Joint attention + scanpath entropy | 0.60–0.70 |
| Speech only | Repair rate + F0 variability | 0.55–0.65 |
| **W_team** | Full composite | **0.75–0.85** |

Likelihood ratio tests for nested comparisons. Report AUC-ROC with 95% CI from bootstrap. **Not built.**

---

## 9. Expected Results

### 9.1 High Confidence (established in literature)

- HR synchrony (%DET) significantly higher in passing trials (PNAS 2024 found >70% prediction accuracy from HR alone)
- NASA-TLX correlates negatively with performance (decades of evidence)
- REPAIR dialogue acts more frequent in failing trials (Clark & Wilkes-Gibbs 1986)

### 9.2 Medium Confidence (supported by recent work)

- W_team explains 15–30% additional variance beyond TLX alone
- Pre-failure windows show %DET drop starting ~20–30 s before error (van Eijndhoven 2025 found physiological markers of coordination transitions)
- Gaze entropy spikes in pre-failure windows (Kiefer et al. 2022 — gaze-based cognitive load in spatial tasks)

### 9.3 Novel Claims (the paper's contribution, untested)

- W_team(t) outperforms all single modalities as a real-time predictor
- The specific temporal cascade: desynchronization → entropy spike → W_team rise → overt error
- SMM → synchrony → performance mediation holds with multimodal process measures
- Gini (workload inequity) adds predictive value beyond mean workload

---

## 10. Control-Room Translation

| Map Task | Control Room |
|----------|-------------|
| Landmark triage + path planning | Alarm triage + procedure selection |
| Instruction → confirmation → draw | Command → read-back → manipulate control |
| Timing pressure (210 s) | Timing pressure (safety margins) |
| IoU / spatial error | Safety-state recovery, procedure deviations |
| W_team(t) → early warning | Same index on HMI micro-world → operator alert |

Phase 2: Browser HMI micro-world (alarms, trends, P&IDs). Phase 3: 3–4 person teams.

---

## 11. Risks and Limitations

| Risk | Mitigation |
|------|-----------|
| HR from wearable BPM (not ECG IBI) has lower precision | Validated for group-level analyses; report as limitation. BPM→IBI conversion standard. |
| 210 s trials borderline for frequency-domain HRV | Report with caveat. Time-domain and nonlinear metrics validated for ≥60 s (Gu et al. 2023). |
| W_team weights overfitted to pilot data | LOO-dyad cross-validation; report generalization performance |
| Pre-failure windows require sufficient failure trials | Pilot calibrates difficulty to ~70–80% pass rate; need ≥30 failure trials for power |
| LF/HF ratio discredited as sympathovagal balance measure | Do not interpret LF/HF; report absolute LF and HF (Billman 2013) |
| Surrogate testing essential for any synchrony claim | 20 time-shift surrogates + z-score significance implemented (Tschacher & Meier 2021) |

---

## 12. What Needs to Be Built

| Component | Priority | Status | Effort |
|-----------|----------|--------|--------|
| **Run pilot (4–6 dyads)** | Critical | Not started | Lab time |
| **`workload.py`** — W_i(t), W_team(t), Gini, EWMA | High | Not built | Medium |
| **`label_pass_fail()`** — binary outcome | High | Not built | Low |
| **Weight calibration** (ridge regression, LOO-CV) | High | Not built | Medium |
| **Mixed-effects models** (R/Python) | High | Not built | Medium |
| **`event_aligned.py`** — pre-failure windows | High | Not built | Medium |
| **Model comparison + ROC** | Medium | Not built | Low |
| **Mediation analysis** | Medium | Not built | Low |
| **SMM similarity score** from PSMM | Medium | Not built | Low |
| **Run full study (30 dyads)** | Critical | Not started | Major |

---

## 13. References

### Physiological Synchrony
- Tognoli, E., et al. (2024). Interpersonal heart rate synchrony predicts effective information processing in a naturalistic group decision-making task. *PNAS*, 121(21), e2313801121. https://www.pnas.org/doi/10.1073/pnas.2313801121
- van Eijndhoven, K. H. J., et al. (2025). Team coordination breakdowns: Examining physiological features underlying transitions in coordination dynamics. *Group Dynamics*, 15553434251328803. https://journals.sagepub.com/doi/10.1177/15553434251328803
- Interpersonal physiological synchrony during dyadic joint action. (2025). *Psychophysiology*. https://pmc.ncbi.nlm.nih.gov/articles/PMC11913774/
- Correlates of interpersonal physiological synchrony and sources of empirical heterogeneity. (2026). *Nature Reviews Psychology*. https://www.nature.com/articles/s44159-026-00535-4
- Physiological and behavioral synchrony predict group cohesion and performance. (2020). *Scientific Reports*. https://www.nature.com/articles/s41598-020-65670-1
- daSilva, E. B. & Wood, A. (2025). How and why people synchronize: An integrated perspective. *Personality and Social Psychology Review*. https://journals.sagepub.com/doi/10.1177/10888683241252036

### RQA / CRQA / MdRQA
- Wallot, S. & Leonardi, G. (2018). Analyzing multivariate dynamics using CRQA, DCRP, and MdRQA — A tutorial in R. *Frontiers in Psychology*, 9, 2232. https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.02232/full
- Wallot, S., Roepstorff, A. & Monseter, D. (2016). MdRQA for multidimensional time-series. *Frontiers in Psychology*, 7, 1835. https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2016.01835/full
- Tschacher, W. & Meier, D. (2021). Beyond dyadic coupling: mv-SUSY. *Entropy*. https://pmc.ncbi.nlm.nih.gov/articles/PMC8623376/

### HRV
- Gu, Y., et al. (2023). Effectiveness of time domain and nonlinear HRV metrics in ultra-short time series. *Physiological Reports*. https://physoc.onlinelibrary.wiley.com/doi/10.14814/phy2.15863
- Billman, G. E. (2013). The LF/HF ratio does not accurately measure cardiac sympatho-vagal balance. *Frontiers in Physiology*. https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2013.00026/full
- PMC (2024). HRV standardization review. https://pmc.ncbi.nlm.nih.gov/articles/PMC11439429/

### Workload
- Hart, S. G. & Staveland, L. E. (1988). Development of NASA-TLX. *Advances in Psychology*, 52, 139–183.
- Lucchese, et al. (2025). Comprehensive systematic literature review on cognitive workload. *IET Collaborative Intelligent Manufacturing*. https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cim2.70025

### Eye Tracking
- Richardson, D. C. & Dale, R. (2005). Looking to understand: The coupling between speakers' and listeners' eye movements. *Cognitive Science*. https://doi.org/10.1207/s15516709cog0000_29
- Schneider, B. & Pea, R. (2013). Real-time mutual gaze perception. *CSCL*. https://doi.org/10.1007/s11412-013-9178-1
- Shiferaw, B., et al. (2019). Review of gaze entropy measures. *Behavior Research Methods*. https://doi.org/10.3758/s13428-019-01226-0
- Krejtz, K., et al. (2014). Gaze transition entropy. *ETRA*. https://doi.org/10.1145/2578153.2578176
- Schneider, B., et al. (2024). Dual eye tracking and collaborative learning: A systematic review. *Educational Research Review*. https://doi.org/10.1016/j.edurev.2024.100598

### Map Task & Shared Mental Models
- Anderson, A. H., et al. (1991). The HCRC Map Task Corpus. *Language and Speech*, 34(4), 351–366.
- Clark, H. H. & Wilkes-Gibbs, D. (1986). Referring as a collaborative process. *Cognition*, 22(1), 1–39.
- DeChurch, L. A. & Mesmer-Magnus, J. R. (2010). Measuring shared mental models: Meta-analysis. *Group Dynamics*, 14(1), 1–14.
- Mathieu, J. E., et al. (2000). The influence of shared mental models on team process and performance. *Journal of Applied Psychology*, 85(2), 273–283.
