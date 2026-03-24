# Paper 1: Real-Time Multimodal Prediction of Coordination Failures in Dyadic Collaboration

## Target Venue
Nature Human Behaviour / PNAS / Psychophysiology

---

## 1. Central Thesis

Coordination failures in dyadic spatial tasks are preceded by detectable multimodal signatures — drops in physiological synchrony, gaze desynchronization, speech disfluency, and rising workload — that emerge 10–60 seconds before overt errors. We introduce a live Team Workload Index, W_team(t), integrating HR, gaze, speech, and subjective load, and demonstrate that it predicts trial-level pass/fail and moment-to-moment accuracy better than any single modality alone. The method generalizes to high-stakes operator settings (nuclear/chemical control rooms).

---

## 2. Research Questions

**RQ1**: Can coordination failures be predicted from multimodal physiological and behavioral signals before they manifest as task errors?

**RQ2**: Does the proposed live Team Workload Index (W_team) outperform single-modality predictors of task performance?

**RQ3**: Are pre-failure windows (−60s to 0s before error onset) characterized by distinct multimodal signatures compared to successful segments?

**RQ4**: Does physiological synchrony (HR %DET, gaze joint attention) mediate the relationship between shared mental models and task performance?

---

## 3. Hypotheses

**H1**: Higher HR synchrony (%DET_HR) and gaze synchrony (%DET_gaze) predict better task accuracy (lower spatial deviation, higher IoU) and fewer repair episodes.

**H2**: Pre-failure windows show:
- (a) HR %DET drops 10–30 s before large errors
- (b) Gaze entropy spikes (scanpath entropy increase)
- (c) Speech overlap increases and repair acts increase
- (d) W_team(t) rises preceding the failure

**H3**: W_team(t) — combining HR quotient, HRV (RMSSD), gaze entropy, speech stress, and TLX — explains significantly more variance in performance than NASA-TLX alone or any single physiological channel.

**H4**: Structural SMM similarity → process synchrony (HR + gaze %DET) → performance, with synchrony partially mediating the SMM–performance link.

---

## 4. Experimental Design

### 4.1 Task
Director–Matcher Map Task. 8 trials (2 warmup + 6 data), 210 s each, role swap mid-session. Maps have subtly different landmarks between roles.

### 4.2 Participants
Target: 30–40 dyads (60–80 participants). Power analysis: with 6 data trials per dyad = 180–240 trial-level observations. Mixed-effects models with random intercepts for dyad achieve adequate power at N ≥ 25 dyads for medium effects (d = 0.5).

### 4.3 Data Captured Per Trial

| Modality | Source | Rate | Features |
|----------|--------|------|----------|
| Heart rate | Galaxy Watch 4 → WS → backend | ~1 Hz (BPM) | HR quotient, RMSSD, SDNN, ln(RMSSD), pNN50, SampEn, DFA α1, SD1/SD2, HF power |
| Eye tracking | SmartEye Pro 10 or Aurora (iMotions) | 60 Hz | Fixations, saccades, pupil, scanpath entropy, AOI dwell, joint attention |
| Speech | Laptop mic → WebM/Opus | Continuous | F0, RMS, speech rate, pauses, turn-taking, dialogue acts, repair count |
| Drawing behavior | Canvas strokes | Continuous | Stroke kinematics, hesitations, erase ratio, coverage growth |
| Subjective workload | NASA-TLX | Post-trial | 6 dimensions, 0–100 |
| Perceived SMM | PSMM survey | Post-trial | 8 items, 1–7 Likert (task + team factors) |

---

## 5. Outcome Variables

### 5.1 Trial-Level (Binary: Pass/Fail)

A trial passes if ALL of the following are met:
- Completed within 210 s (no timeout)
- Drawing accuracy IoU ≥ threshold (pilot-calibrated, e.g., IoU ≥ 0.15)
- Boundary F1 ≥ threshold (e.g., ≥ 0.20)
- Director rates target_reached = "yes"

Fail = timeout OR below accuracy thresholds OR Director rates target_reached = "no".

**What exists**: IoU, F1, SSIM, Hausdorff, Chamfer, boundary F1 all computed in postprocess.py. Director's `trial_success` form captures `target_reached` and `path_confidence`.

**What needs building**: A `label_pass_fail()` function that combines these into a single binary outcome with configurable thresholds. To be calibrated in pilot (aim for 70–80% pass rate).

### 5.2 Trial-Level (Continuous)

- Time to complete (or remaining time at trial end)
- IoU / boundary F1 / Chamfer distance (continuous accuracy)
- Number of repair dialogue acts (from llm_eval.py)
- NASA-TLX composite score

### 5.3 Moment-to-Moment (Time Series)

- W_i(t): individual workload index (1 Hz)
- W_team(t): team workload index (1 Hz)
- Time-series drawing accuracy (IoU at every 5th stroke)

---

## 6. The Live Workload Index

### 6.1 Individual Workload W_i(t)

From the study design (slide 7):

```
W_i(t) = α₁·z(HRQ_i) + α₂·z(−RMSSD_i) + α₃·z(GazeEnt_i) + α₄·z(SpeechStress_i) + α₅·z(TLX_i,block) + α₆·z(EEG_θ/β,i)
```

For this study (no EEG):

```
W_i(t) = α₁·z(HRQ_i) + α₂·z(−RMSSD_i) + α₃·z(GazeEnt_i) + α₄·z(SpeechStress_i) + α₅·z(TLX_i,block)
```

Where:
- `HRQ = (HR − HR_baseline) / HR_baseline` — heart rate quotient (higher = more aroused)
- `RMSSD` — negated because lower RMSSD = higher sympathetic load
- `GazeEnt` — scanpath entropy (higher = less efficient visual search)
- `SpeechStress` — composite of F0 slope, intensity, speech rate
- `TLX` — block-level subjective load (held constant within block, updated at each TLX administration)
- `z(·)` — within-person z-score standardization
- `α_k` — weights, initialized equal (1/5), calibrated in pilot via regression against performance

**Smoothing**: Exponentially Weighted Moving Average (EWMA), span = 5 s, updated at 1 Hz.

### 6.2 Team Workload W_team(t)

```
W_team(t) = β₁·W̄_i(t) + β₂·Gini(W_i) + β₃·z(−%DET_HR) + β₄·z(−JointAttn)
```

Where:
- `W̄_i` — mean of Director and Matcher individual workloads
- `Gini(W_i)` — workload inequity (0 = equal, 1 = one person doing all work)
- `%DET_HR` — HR synchrony determinism (from windowed CRQA)
- `JointAttn` — proportion of time both participants fixate same AOI (2 s tolerance)
- `β_k` — initialized equal, calibrated via regression to predict time/error/pass-fail

**What exists**: All input features are computed in postprocess.py (HR stats, gaze features, prosody). Windowed CRQA exists. Joint attention exists in gaze_pair.csv.

**What needs building**:
1. `workload.py` — compute W_i(t) and W_team(t) from the aligned time-series data
2. EWMA smoothing at 1 Hz
3. Gini coefficient computation for workload inequity
4. Weight calibration procedure (ridge regression with cross-validation)

---

## 7. Analyses

### 7.1 Analysis 1: Trial-Level Prediction (Mixed-Effects)

**Model**:
```
Performance ~ Structural_SMM + %DET_HR + NASA_TLX + Repairs + W_team_mean + (1|Dyad)
```

Where `Performance` = IoU (continuous) or Pass/Fail (logistic).

**What exists**: All predictors computed. PSMM scores captured.

**What needs building**: R or Python script using `lme4` (R) or `statsmodels` (Python) for mixed-effects models. Need to compute `Structural_SMM` as the similarity between Director's and Matcher's PSMM responses (e.g., profile correlation or absolute difference score).

### 7.2 Analysis 2: Event-Aligned Pre-Failure Windows

**Approach**:
1. Identify failure events: trial_end with IoU below threshold, or first repair dialogue act
2. Extract −60s to 0s window before each failure event
3. Extract matched −60s windows from successful trials (control)
4. Compare HR/gaze/speech/W_team trajectories between failure and control windows

**Metrics in windows**:
- HR: mean BPM, RMSSD, %DET_HR (windowed CRQA at 30s)
- Gaze: scanpath entropy, joint attention rate, fixation duration
- Speech: pause rate, overlap %, repair act count, F0 variability
- W_team: trajectory (rising = approaching overload?)

**Statistical test**: Growth curve models or permutation tests comparing failure vs. control window trajectories.

**What exists**: Windowed CRQA (60s window, 30s step). Dialogue act classification (REPAIR detection). All per-trial features.

**What needs building**:
1. `event_aligned.py` — window extraction aligned to failure onset
2. Failure event detection (IoU drop below threshold at each stroke, or repair act timestamp)
3. Trajectory comparison code (growth curves or cluster-based permutation tests)

### 7.3 Analysis 3: Mediation

**Path model**:
```
Structural SMM → Process synchrony (HR %DET + gaze joint attention) → Performance (IoU)
```

Test: Does process synchrony partially mediate the SMM → Performance relationship?

**What exists**: All variables computed.

**What needs building**: Mediation analysis using `lavaan` (R) or `pingouin`/`statsmodels` (Python). Baron & Kenny steps + Sobel test + bootstrapped confidence intervals.

### 7.4 Analysis 4: W_team Validation

Compare predictive power of:
1. NASA-TLX alone
2. HR alone (%DET_HR)
3. Gaze alone (joint attention)
4. Speech alone (repair rate)
5. W_team(t) (multimodal composite)

**Method**: Nested model comparison (likelihood ratio tests) predicting pass/fail. Report AUC-ROC for each.

**What needs building**: Model comparison script with ROC analysis.

---

## 8. Expected Results

### 8.1 Primary (high confidence)

- HR synchrony (%DET) will be significantly higher in passing trials than failing trials (based on PNAS 2024, Psychophysiology 2025 findings)
- NASA-TLX will correlate with performance (well-established)
- Repair rate will be higher in failing trials

### 8.2 Secondary (medium confidence)

- W_team will explain 15–30% more variance than TLX alone
- Pre-failure windows will show %DET drop starting ~20–30s before error onset (based on van Eijndhoven et al. 2025)
- Gaze entropy will spike in pre-failure windows (less efficient search)

### 8.3 Novel (the paper's contribution)

- W_team(t) as a real-time composite outperforms all single modalities
- The specific temporal signature: desynchronization → entropy spike → W_team rise → overt error
- Structural SMM → synchrony → performance mediation path holds

---

## 9. Control-Room Translation Argument

| Map Task | Control Room |
|----------|-------------|
| Landmark triage + path planning | Alarm triage + procedure selection |
| Instruction → confirmation → draw | Command → read-back → manipulate control |
| Timing pressure (210s) | Timing pressure (safety margins) |
| IoU / spatial error | Safety-state recovery, procedure deviations |
| W_team(t) | Same index on HMI micro-world |

The paper argues: if W_team predicts failures in map tasks, the same index — with identical analytics — can provide early warning in control rooms. Phase 2 (browser HMI micro-world) and Phase 3 (3–4 person teams) extend this.

---

## 10. What Needs to Be Built

| Component | Priority | Status | Effort |
|-----------|----------|--------|--------|
| **Run pilot sessions (4–6 dyads)** | Critical | Not started | Lab time |
| **`label_pass_fail()`** — binary outcome labeling | High | Not built | Low (1 function) |
| **`workload.py`** — W_i(t) and W_team(t) computation | High | Not built | Medium (core contribution) |
| **Weight calibration** — ridge regression for α_k, β_k | High | Not built | Medium |
| **Mixed-effects models** — R/Python script | High | Not built | Medium |
| **`event_aligned.py`** — pre-failure window extraction | High | Not built | Medium |
| **Mediation analysis** — lavaan or statsmodels | Medium | Not built | Low |
| **W_team validation** — model comparison + ROC | Medium | Not built | Low |
| **Run full study (30–40 dyads)** | Critical | Not started | Major lab effort |
| **Structural SMM similarity score** from PSMM | Medium | Not built | Low (profile correlation) |

---

## 11. Key References

- Anderson et al. (1991) — HCRC Map Task Corpus
- Hart & Staveland (1988) — NASA-TLX
- DeChurch & Mesmer-Magnus (2010) — Measuring Shared Mental Models meta-analysis
- Wallot & Leonardi (2018) — CRQA/DCRP/MdRQA tutorial
- PNAS (2024) — HR synchrony predicts group decision-making
- van Eijndhoven et al. (2025) — Team coordination breakdowns via physiological features
- Nature Reviews Psychology (2026) — Interpersonal physiological synchrony review
- Billman (2013) — LF/HF ratio critique
- Tschacher & Meier (2021) — mv-SUSY surrogate framework
