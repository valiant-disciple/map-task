# Literature Review: HRV, RQA, and Interpersonal Physiological Synchrony in Dyadic Collaboration

## 1. Time-Domain HRV Metrics

Standard per-individual measures computed from inter-beat intervals (IBI).

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| RMSSD | Root mean square of successive IBI differences | Primary short-term parasympathetic/vagal marker. Valid even in ultra-short recordings (≥60s). Preferred over pNN50. |
| ln(RMSSD) | Log-transformed RMSSD | Normalizes distribution; standard in most publications |
| SDNN | Standard deviation of NN intervals | Overall HRV reflecting all cyclic components. Standard requires 5 min but validated down to ~60s |
| pNN50 | % successive intervals differing >50ms | Parasympathetic marker; highly correlated with RMSSD |
| SDSD | SD of successive differences | Equivalent to RMSSD for stationary series |
| Mean RR / Mean HR | Average inter-beat interval / heart rate | Baseline measure for normalization |

**Key reference**: Gu et al. (2023) validated RMSSD and SampEn for ultra-short time series. [Link](https://physoc.onlinelibrary.wiley.com/doi/10.14814/phy2.15863)

**Best practice**: PMC (2024) standardization review — [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC11439429/)

---

## 2. Frequency-Domain HRV

| Metric | Band | Interpretation | Status |
|--------|------|----------------|--------|
| HF Power | 0.15–0.40 Hz | Parasympathetic/vagal. Most reliable frequency metric | ✓ Standard |
| LF Power | 0.04–0.15 Hz | Mixed sympathetic + parasympathetic + baroreflex | ⚠ NOT purely sympathetic |
| LF/HF Ratio | — | Purported "sympathovagal balance" | ✗ Largely discredited (Billman, 2013) |
| VLF Power | 0.003–0.04 Hz | Very low frequency | Requires >5 min recordings |
| Total Power | 0–0.40 Hz | Overall spectral power | ✓ Standard |

**Critical caveat**: LF/HF ratio does NOT accurately measure sympathovagal balance. LF is modulated by both ANS branches and baroreflexes. Report absolute LF and HF. For 210-second trials, frequency-domain is borderline (below 5-min standard) but defensible with appropriate caveats.

**Key reference**: Billman (2013) — "The LF/HF ratio does not accurately measure cardiac sympatho-vagal balance." *Frontiers in Physiology*. [Link](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2013.00026/full)

---

## 3. Nonlinear HRV

| Metric | Description | Notes |
|--------|-------------|-------|
| Sample Entropy (SampEn) | Regularity/complexity of IBI series | More reliable than ApEn; lower = more regular. Validated for ultra-short recordings. |
| DFA alpha1 | Short-term fractal scaling exponent (4–16 beats) | Captures fractal correlation; α1 ≈ 1.0 in healthy individuals. Valid for short recordings. |
| SD1 (Poincaré) | Short-term beat-to-beat variability | Mathematically ≈ RMSSD/√2. Parasympathetic marker. |
| SD2 (Poincaré) | Long-term variability | Related to SDNN. |
| SD1/SD2 | Poincaré ratio | Correlates with DFA alpha1; captures sympathovagal dynamics |

**Recommended**: SampEn, DFA alpha1, SD1/SD2 — validated for short recordings and capture complexity beyond linear metrics.

---

## 4. Recurrence Quantification Analysis (RQA)

Auto-recurrence (per individual) and cross-recurrence (between dyad members). Computed from recurrence plots of the IBI time series.

### RQA Metrics

| Metric | Abbreviation | Description |
|--------|-------------|-------------|
| Recurrence Rate | %REC / RR | Proportion of recurrent points; higher = more self-similar |
| Determinism | DET | % recurrent points on diagonal lines; higher = more predictable |
| Average Diagonal Line | L / ADL | Mean diagonal line length; average prediction time |
| Longest Diagonal Line | L_max | Max diagonal; inversely related to largest Lyapunov exponent |
| Divergence | DIV | 1/L_max; rate of trajectory divergence |
| Entropy (diagonal) | ENTR | Shannon entropy of diagonal line distribution; complexity of deterministic structure |
| Laminarity | LAM | % recurrent points on vertical lines; intermittency/laminar states |
| Trapping Time | TT | Mean vertical line length; how long system stays in a state |
| Longest Vertical Line | V_max | Maximum vertical line length |
| Entropy (vertical) | V_entr | Shannon entropy of vertical lines |

### Typical Parameters for HR/IBI Data

| Parameter | Method | Typical Range |
|-----------|--------|---------------|
| Time delay (τ) | Average Mutual Information — first local minimum | 1–5 for IBI at ~1–4 Hz |
| Embedding dimension (m) | False Nearest Neighbors — where FNN% → 0 | 1–5 for IBI |
| Radius (r) | Target ~2–5% recurrence rate | Data-dependent |
| Norm | Euclidean (standard) | — |
| Min line length (l_min) | Convention | 2 |
| Theiler window | Exclude LoI neighbors | 1 (0 for CRQA) |

**Key reference**: Wallot & Leonardi (2018) — "Analyzing Multivariate Dynamics Using CRQA, DCRP, and MdRQA." *Frontiers in Psychology*. [Link](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.02232/full)

---

## 5. Cross-Recurrence Quantification Analysis (CRQA)

Same metric set as RQA but from the cross-recurrence plot of two individuals' time series. Captures coupling/synchrony.

| CRQA Metric | Dyadic Interpretation |
|-------------|----------------------|
| RR (cross) | Overall physiological coupling; how often both systems visit similar states |
| DET (cross) | Predictability of coupling; sustained coordination vs. incidental overlap |
| L (cross) | Average duration of coordinated epochs |
| L_max (cross) | Longest sustained coordination episode |
| ENTR (cross) | Complexity of coupling structure |
| LAM (cross) | One system "waiting" for the other |
| TT (cross) | Duration of trapping/waiting episodes |

**Key reference**: Coco et al. (2021) — "Unidimensional and Multi-dimensional Methods for RQA with crqa." *The R Journal*.

---

## 6. Multidimensional RQA (MdRQA)

Embeds N time series in a single joint phase space, producing a single recurrence plot for the entire system.

| | CRQA | MdRQA |
|--|------|-------|
| Input | Two univariate time series | N series in one phase space |
| Captures | Pairwise coupling | Joint system-level dynamics |
| Use case | Dyadic synchrony | Holistic coordination; multivariate (e.g., HR + EDA) |
| Output | Cross-recurrence metrics | Same metrics but reflecting joint recurrence |

**When to use**: When you want a single measure of system-level coordination, or when analyzing multiple channels jointly.

**Key reference**: Wallot, Roepstorff & Monseter (2016) — "MdRQA for Multidimensional Time-Series." *Frontiers in Psychology*. [Link](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2016.01835/full)

**Software**: [PyMdRQA](https://github.com/furmanlukasz/PyMdRQA), R `crqa` package, [MATLAB MdRQA toolbox](https://github.com/Wallot/MdRQA)

---

## 7. Diagonal Cross-Recurrence Profile (DCRP)

Computes %REC along diagonals offset from the Line of Synchrony (LoS) in the cross-recurrence plot.

- **Peak lag**: which participant leads physiologically
- **Profile width**: coupling flexibility (narrow = rigid synchrony, wide = flexible)
- **LoS %REC**: in-phase synchrony

Particularly relevant for director-matcher tasks — reveals whether Director's HR changes precede or follow Matcher's.

**Key reference**: Wallot & Leonardi (2018) — same paper as RQA tutorial above.

---

## 8. Windowed / Time-Varying Approaches

| Method | Description | Use Case |
|--------|-------------|----------|
| Windowed CRQA | CRQA in sliding windows | Track synchrony evolution within trial |
| Windowed Cross-Correlation | Pearson r in sliding windows | Linear coupling over time; peak r and lag |
| DCRP over time | Not standard; compute DCRP in windows | Leadership dynamics evolving |

**Trend metric**: Slope of windowed CRQA RR/DET over time — does synchrony build up during the trial?

---

## 9. Complementary Synchrony Measures

| Method | Type | Captures | Complements |
|--------|------|----------|-------------|
| Windowed Cross-Correlation | Linear, time-domain | Linear time-lagged correlation | CRQA (adds linear perspective) |
| Transfer Entropy | Information-theoretic | Directed information flow (A→B vs B→A) | DCRP (directional coupling) |
| Symbolic Transfer Entropy | Info-theoretic, robust | Simplified TE using symbolic dynamics | More robust for short noisy data |
| Wavelet Coherence | Time-frequency | Frequency-specific coupling over time | Shows WHICH frequency bands sync |
| Dynamic Time Warping | Alignment-based | Shape similarity despite warping | Compare HR trajectory shapes across dyads |
| Phase Synchronization | Nonlinear | Phase locking of oscillatory components | Respiratory-cardiac coupling |
| Granger Causality | Linear, model-based | Linear directional prediction | Simple baseline for directed coupling |
| Surrogate/Pseudo-dyad testing | Statistical framework | Chance-level baseline for all metrics | **Non-negotiable** for any synchrony paper |

**Recommended pairing**: CRQA (nonlinear) + Windowed Cross-Correlation (linear) + Transfer Entropy (directional) + Surrogate testing (significance)

---

## 10. Surrogate Testing

**Critical for publication**: Real-dyad CRQA/MdRQA values must be compared against surrogate/pseudo-dyad baselines.

**Methods**:
- **Time-shift surrogates**: Circularly shift one participant's time series by random amount, recompute CRQA. Repeat 20–100 times.
- **Participant shuffling**: Pair participants from different dyads. Compute CRQA on mismatched pairs.
- **mv-SUSY** (Tschacher & Meier, 2021): Multivariate Surrogate Synchrony framework for systematic testing.

**Report**: Real-dyad metric, surrogate mean ± SD, z-score = (real − surr_mean) / surr_SD.

**Key reference**: Tschacher & Meier (2021) — "Beyond Dyadic Coupling: mv-SUSY." *Entropy*. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC8623376/)

---

## 11. Key Recent Papers (2022–2026)

### Reviews & Meta-analyses

- **Nature Reviews Psychology (2026)** — "Correlates of interpersonal physiological synchrony and sources of empirical heterogeneity." Comprehensive synthesis of IPS findings. [Link](https://www.nature.com/articles/s44159-026-00535-4)

- **daSilva & Wood (2025)** — "How and Why People Synchronize: An Integrated Perspective." Broad review of synchrony mechanisms. [Link](https://journals.sagepub.com/doi/10.1177/10888683241252036)

- **Oaepublish (2025)** — "Exploring cardiac physiological synchrony and its implications for stress and anxiety." Recent review of cardiac synchrony methods. [Link](https://www.oaepublish.com/articles/and.2025.14)

### Empirical Studies

- **PNAS (2024)** — "Interpersonal heart rate synchrony predicts effective information processing in a naturalistic group decision-making task." HR synchrony → group performance. [Link](https://www.pnas.org/doi/10.1073/pnas.2313801121)

- **Psychophysiology (2025)** — "Interpersonal Physiological Synchrony During Dyadic Joint Action." CRQA on HR during joint tasks. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC11913774/)

- **van Eijndhoven et al. (2025)** — "Team Coordination Breakdowns: Examining Physiological Features." Windowed MdRQA for team HR/EDA. [Link](https://journals.sagepub.com/doi/10.1177/15553434251328803)

- **Nature Scientific Reports (2023)** — "How our hearts beat together." Cardiac coupling in collaborative contexts. [Link](https://www.nature.com/articles/s41598-023-39083-9)

### Methodology

- **Marwan & Webber (2023)** — "Trends in Recurrence Analysis of Dynamical Systems." Latest RQA methodological advances. *European Physical Journal Special Topics*. [Link](https://link.springer.com/article/10.1140/epjs/s11734-023-00766-z)

- **Gu et al. (2023)** — "Effectiveness of time domain and nonlinear HRV metrics in ultra-short time series." Validates RMSSD and SampEn for short recordings. [Link](https://physoc.onlinelibrary.wiley.com/doi/10.14814/phy2.15863)

- **Frontiers (2023)** — "Inter-system recurrence networks for physiological coupling." Novel network approach. [Link](https://www.frontiersin.org/journals/network-physiology/articles/10.3389/fnetp.2023.1289983/full)

### Foundational Tutorials (Still Essential)

- **Wallot & Leonardi (2018)** — "Analyzing Multivariate Dynamics Using CRQA, DCRP, and MdRQA — A Tutorial in R." *Frontiers in Psychology*. The definitive implementation tutorial. [Link](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.02232/full)

- **Coco et al. (2021)** — "Unidimensional and Multi-dimensional Methods for RQA with crqa." *The R Journal*. R `crqa` package documentation.

- **Wallot, Roepstorff & Monseter (2016)** — "MdRQA for Multidimensional Time-Series." *Frontiers in Psychology*. Original MdRQA paper. [Link](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2016.01835/full)

---

## 12. Implementation Notes for This Study

- **Trial duration (210s)**: Sufficient for all time-domain, nonlinear, and RQA/CRQA metrics. Borderline for frequency-domain (LF needs ≥2 min; HF needs ≥1 min).
- **HR from wearable (BPM, not IBI)**: Convert BPM → IBI via `IBI_ms = 60000 / BPM`. Precision is lower than ECG-derived IBI but validated for group-level analyses.
- **Preprocessing**: Resample IBI to uniform 4 Hz (cubic spline interpolation) before spectral and RQA analysis.
- **Parameter optimization**: Use AMI for delay, FNN for embedding dimension, target 2–5% recurrence for radius.
- **Surrogate testing**: Always compare real vs pseudo-dyad CRQA. Report z-scores.
- **Software**: PyRQA (Python), `crqa` (R), PyMdRQA (Python). Our pipeline uses PyRQA with custom MdRQA implementation.
