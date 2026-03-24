# Literature Review: Eye Tracking, Gaze Features, and Dyadic Gaze Coordination in Collaborative Spatial Tasks

## 1. Fixation Metrics

Standard per-individual measures computed from fixation events detected by the eye tracker.

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Fixation Count | Total number of fixations per trial | More fixations = more exploratory scanning or greater task difficulty |
| Fixation Duration (mean) | Average fixation duration (ms) | Longer = deeper cognitive processing. ~200-300ms typical |
| Fixation Duration (median) | Median duration; robust to outlier fixations | Preferred over mean for skewed distributions |
| Fixation Duration (SD) | Variability of fixation durations | High variability = alternating fast scanning and deep processing |
| Fixation Duration (max) | Longest single fixation | >800ms may indicate confusion or loss of tracking |
| Fixation Rate | Fixations per second (Hz) | General scanning tempo; ~3-4 Hz typical |
| Fixation Dispersion | Spatial spread of fixation centroids (SD of x,y or convex hull area) | Wider = more exploratory; narrower = more focused |
| Ambient/Focal Ratio | Ratio of short (<150ms, "ambient") to long (>300ms, "focal") fixations | Based on two-mode visual processing model |
| Fixation-to-Saccade Ratio | Proportion of time in fixations vs. saccades | Higher = more processing relative to scanning |

**Key references**:
- Holmqvist et al. (2011) — *Eye Tracking: A Comprehensive Guide to Methods and Measures* — foundational reference for all basic metrics
- Henderson (2003) — "Human gaze control during real-world scene viewing" — fixation duration and cognitive processing
- Velichkovsky et al. (2002) — Ambient vs. focal processing modes [Link](https://link.springer.com/chapter/10.1007/3-540-36181-2_60)
- Goldberg & Kotval (1999) — "Computer interface evaluation using eye movements" — fixation rate, dispersion [Link](https://doi.org/10.1016/S1071-5819(99)80058-4)

---

## 2. Saccade Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Saccade Count | Total saccades per trial | General scanning activity |
| Saccade Amplitude (mean, SD) | Average angular/pixel distance per saccade | Large = global scanning; small = local inspection |
| Saccade Peak Velocity (mean) | Maximum speed during saccade (deg/s or px/s) | Follows "main sequence"; deviations indicate fatigue/load |
| Saccade Duration (mean) | Average temporal length (ms) | Correlated with amplitude |
| Saccade Direction Distribution | Proportion in angular bins (up/down/left/right) | Reveals directional scanning biases |
| Regressive Saccade Rate | Proportion of backward/re-tracing saccades | Indicates re-checking or error detection |
| Saccade Main Sequence Slope | Slope of amplitude vs. peak velocity regression | Deviation from ~500 deg/s at 20 deg may indicate fatigue |

**Key references**:
- Bahill et al. (1975) — "The main sequence, a tool for studying human eye movements" [Link](https://doi.org/10.1016/0025-5564(75)90075-9)
- Di Stasi et al. (2013) — Saccade velocity and cognitive fatigue [Link](https://doi.org/10.1016/j.neubiorev.2013.01.023)
- Rayner (1998) — "Eye movements in reading and information processing: 20 years of research" [Link](https://doi.org/10.1037/0033-2909.124.3.372)

---

## 3. Pupil Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Mean Pupil Diameter | Average pupil size (mm) over trial | Baseline arousal/luminance indicator |
| Pupil Diameter SD | Variability of pupil size | Fluctuating cognitive demands |
| Baseline-Corrected Pupil | Diameter minus pre-trial baseline mean | Isolates task-evoked dilation from tonic arousal |
| TEPR | Task-Evoked Pupillary Response: peak dilation relative to baseline | Classic cognitive effort index |
| Pupil Dilation Rate | Derivative of pupil size (mm/s) | Speed of cognitive engagement onset |
| ICA | Index of Cognitive Activity: frequency of rapid small dilations via wavelet decomposition | Real-time cognitive load index |
| Low-Frequency Fluctuations | Power in 0.04-0.15 Hz band | Sustained attention and arousal fluctuations |

**Critical caveat**: Pupil size is affected by luminance. Since both participants view maps with similar brightness, within-session comparisons are valid, but absolute values require luminance control or baseline correction.

**Key references**:
- Beatty (1982) — "Task-evoked pupillary responses, processing load, and the structure of processing resources" [Link](https://doi.org/10.1037/0033-2909.91.2.276)
- Kahneman (1973) — *Attention and Effort* — foundational pupillometry work
- Mathot et al. (2018) — "Safe and sensible preprocessing and baseline correction of pupil-size data" [Link](https://doi.org/10.3758/s13428-017-1007-2)
- Marshall (2002) — The Index of Cognitive Activity [Link](https://doi.org/10.1207/S15327590IJHC1401_3)
- Klingner et al. (2008) — Pupil variability and cognitive load [Link](https://doi.org/10.1145/1344471.1344557)

---

## 4. Blink Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Blink Rate | Blinks per minute | Increases with fatigue; decreases with high visual demand/engagement |
| Blink Duration (mean) | Average time eyelids closed (ms) | Longer = fatigue, reduced alertness |
| Blink Inhibition | Suppression of blinks during critical processing | Deviation from expected rate signals focused visual attention |

**Data availability**: SmartEye has explicit `Blink` column. Aurora has `ET_EyelidOpeningLeft/Right` — blinks detected when eyelid opening drops below threshold (~0.2mm).

**Key references**:
- Stern et al. (1984) — "The endogenous eyeblink" — blink rate and cognitive state
- Caffier et al. (2003) — Blink duration and alertness [Link](https://doi.org/10.1016/S1389-9457(03)00066-1)
- Nakano et al. (2009) — Blink inhibition during visual attention [Link](https://doi.org/10.1098/rspb.2009.0828)

---

## 5. Scanpath Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Scanpath Length | Total Euclidean distance of gaze path (px) | Longer = more visual exploration or inefficient search |
| Convex Hull Area | Area of smallest polygon enclosing all fixation points | Spatial extent of visual exploration |
| Scanpath Entropy (SGE) | Shannon entropy over spatial grid of fixation locations | Low = concentrated; high = distributed/exploratory |
| Gaze Transition Entropy (GTE) | Entropy of AOI-to-AOI transition matrix | Low = systematic scanning; high = random |
| Nearest Neighbor Index | Observed vs. expected mean nearest-neighbor distance | <1 = clustered fixations; >1 = dispersed |
| Scanpath Similarity (MultiMatch) | Multi-dimensional comparison: shape, direction, length, position, duration | Quantitative comparison of two scanpaths |
| RQA of Gaze | Recurrence metrics on gaze time series | Captures deterministic patterns in scanning behavior |

**Key references**:
- Krejtz et al. (2014) — "Gaze transition entropy" [Link](https://doi.org/10.1145/2578153.2578176)
- Shiferaw et al. (2019) — "Gaze entropy and task performance" [Link](https://doi.org/10.3758/s13428-019-01226-0)
- Jarodzka et al. (2010) — "A vector-based, multidimensional scanpath similarity measure" (MultiMatch) [Link](https://doi.org/10.1145/1743666.1743718)
- Anderson et al. (2013) — "A recurrence-based analysis of scanpath dynamics" [Link](https://doi.org/10.3758/s13428-012-0225-0)
- Clark & Evans (1954) — Nearest Neighbor Index (ecological measure applied to eye tracking)

---

## 6. AOI-Based Metrics

Areas of Interest for this study: map region, timer, toolbar, specific landmarks.

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| AOI Dwell Time | Cumulative fixation time within each AOI | Primary selective attention measure |
| AOI Dwell Proportion | % of total viewing time per AOI | Normalized for trial duration differences |
| AOI Fixation Count | Number of fixations per AOI | Attention allocation frequency |
| AOI Transition Matrix | Frequency of gaze transitions between AOI pairs | Systematic scanning strategies |
| Time to First Fixation | Latency from trial onset to first fixation per AOI | Attentional priority |
| AOI Coverage | Proportion of AOIs visited at least once | Exploration completeness |
| Revisit Count | Number of returns to a previously visited AOI | Re-examination, uncertainty |

**AOI boundaries for this study**:
- Director map: (252, 137) → (889, 1017)
- Matcher map: (267, 137) → (904, 1017)
- Timer: (613, 8) → (735, 65)
- Toolbar: role-dependent

**Key references**:
- Holmqvist et al. (2011) — AOI analysis framework
- Jacob & Karn (2003) — "Eye tracking in HCI and usability research" — dwell time methodology [Link](https://doi.org/10.1016/B978-044451020-4/50031-1)

---

## 7. Cross-Participant Gaze Coordination

Dyadic gaze coupling measures between Director and Matcher. Requires normalizing both participants' gaze to a common map coordinate space.

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| CRQA of Gaze | Cross-recurrence on map-normalized gaze positions | How often both visit similar spatial regions across time |
| Gaze Coupling Lag | Temporal offset of peak cross-recurrence | Who leads attention (positive = Matcher follows Director) |
| Joint AOI Fixation | Proportion of time both fixating same region (~2s tolerance) | Convergent attention episodes |
| Gaze Convergence Index | Mean Euclidean distance between map-space gaze points over time | Lower = more convergent attention |
| Leader-Follower Index | Asymmetry in cross-recurrence profile | Who drives shared attention |
| MdCRQA of Gaze | Multivariate cross-recurrence on 2D gaze | Extends single-dimension CRQA to spatial coordinates |

**Critical implementation note**: Director and Matcher view different maps (full vs. ground), so "looking at the same thing" means looking at the same spatial coordinates in map space, not the same screen position. Normalize using AOI bounds.

**Key references**:
- Richardson & Dale (2005) — "Looking to understand: The coupling between speakers' and listeners' eye movements" [Link](https://doi.org/10.1207/s15516709cog0000_29) — foundational paper for dyadic gaze coupling
- Dale & Richardson (2009) — Extending gaze coupling analysis
- Jermann et al. (2011) — "Gaze coordination in collaborative problem solving"
- Schneider & Pea (2013) — "Real-time mutual gaze perception" [Link](https://doi.org/10.1007/s11412-013-9178-1)
- Wallot et al. (2016) — MdRQA for multidimensional time series

---

## 8. Gaze-Speech Alignment

Unique features exploiting synchronization of Director's gaze with speech output. Requires word-level timestamps from Whisper ASR.

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Gaze-Referent Coupling | Distance between Director's gaze and named landmark at utterance time | Tests grounding hypothesis: speakers look at what they describe |
| Gaze-Speech Lead Time | Temporal offset between first fixating a region and first mentioning it | Gaze leads speech by ~800-1000ms in naming tasks; reflects planning |
| Look-Before-Speak Duration | Fixation time on region before describing it | Longer look-ahead = more complex/ambiguous region |
| Gaze-Pause Alignment | Gaze position during speech pauses (>300ms) | Ahead on route = planning; at current point = difficulty |
| Route Preview Distance | How far ahead on route gaze is vs. what's being described | Analogous to driving look-ahead; large preview = fluent instruction |

**Key references**:
- Griffin & Bock (2000) — "What the eyes say about speaking" [Link](https://doi.org/10.1111/1467-9280.00255)
- Meyer et al. (1998) — Gaze-speech coordination in object naming
- Land & Lee (1994) — Gaze-ahead distance in driving (analogous concept)

---

## 9. Gaze-Drawing Alignment (Matcher-Specific)

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Gaze-Cursor Distance | Euclidean distance between gaze and pen/cursor | Visuo-motor coupling tightness |
| Gaze-to-Draw Lag | Time between first fixating a region and first drawing there | Planning horizon before drawing |
| Drawing-Gaze Coherence | Cross-correlation of gaze and drawing trajectory | Tight coupling = gaze-guided; low = memory-based |
| Pre-Draw Scanning | Fixation pattern between drawing bouts | Context scanning vs. self-monitoring vs. route planning |
| Gaze-on-Own-Drawing | Fraction of time fixating previous strokes | Error checking, self-monitoring |
| Gaze at Stroke Onset | Gaze coordinates at moment each stroke begins | On start point = visually guided; elsewhere = memory-based |

**Key references**:
- Sailer et al. (2005) — "Eye-hand coordination during drawing" [Link](https://doi.org/10.1167/5.12.7)
- Tchalenko (2009) — "Segmentation and accuracy in drawing" — gaze-hand coupling in drawing tasks

---

## 10. Multimodal Features (Gaze + Speech + HR)

Features uniquely enabled by having all three data streams time-synchronized.

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Pupil-HR Correlation | Pearson r between pupil and HR time series | Both reflect arousal; dissociation reveals cognitive vs. physical demand |
| Cognitive Load Composite | z-scored: pupil + fixation duration + inv. blink rate + HR change | Multi-channel cognitive effort estimate |
| Gaze-Speech-HR at Difficulty | Triple convergence at high-error segments | More revisits + pauses + pupil dilation + HR increase |
| Pupil Response to Landmark Mention | TEPR time-locked to landmark names | Larger for incorrectly drawn landmarks = prediction difficulty |
| HR Synchrony × Gaze Synchrony | HR coupling during high vs. low gaze convergence | Tests whether attention alignment drives physiological coupling |
| MdCRQA All Modalities | 8D: gaze_x, gaze_y, pupil, HR × 2 participants | Holistic multimodal coupling invisible from single modalities |

**Key references**:
- Konvalinka et al. (2011) — HR synchrony in joint action [Link](https://doi.org/10.1098/rspb.2011.1131)
- Chen et al. (2014) — Multimodal cognitive load assessment
- Palinko et al. (2010) — Combined pupil and driving performance [Link](https://doi.org/10.1145/1743666.1743718)

---

## 11. Landmark and Route Attention

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Landmark Mention-Fixation Concordance | Matcher fixates named landmark within 0-4s of Director mention | Grounding success measure |
| Shared vs. Unique Landmark Attention | Compare attention to landmarks on both vs. one map | Director uses shared landmarks as anchors |
| Landmark Attention Sequence | Edit distance between gaze sequence and route order | Systematic route-following vs. random scanning |
| Director Gaze-Route Distance | Mean distance from Director gaze to nearest GT route point | Visual coupling with route during instruction |
| Route Progress Correlation | Gaze-on-route progress (0-100%) vs. time | Monotonic = linear instructions; non-monotonic = backtracking |

**Key references**:
- Clark & Wilkes-Gibbs (1986) — "Referring as a collaborative process" — common ground theory
- Brown et al. (1984) — "The Map Task: A natural tool for studying collaborative communication" — original HCRC map task design

---

## 12. Key Recent Papers (2022-2026)

### Reviews & Methodology

- **Holmqvist et al. (2023)** — "Eye tracking: empirical foundations for a minimal reporting guideline." *Behavior Research Methods*. [Link](https://doi.org/10.3758/s13428-021-01762-8)

- **Shiferaw et al. (2019)** — "Review of gaze entropy measures for eye-tracking data quality and analysis." [Link](https://doi.org/10.3758/s13428-019-01226-0)

### Dyadic Eye Tracking

- **Schneider et al. (2024)** — "Dual eye tracking and collaborative learning: A systematic review." *Educational Research Review*. [Link](https://doi.org/10.1016/j.edurev.2024.100598)

- **Haataja et al. (2023)** — "Individuals in a group: Metacognitive and regulatory predictors of learning achievement in collaborative learning." *Learning and Individual Differences*. Dual eye tracking + gaze coupling in collaborative problem solving.

- **Skuballa et al. (2022)** — "Dyadic gaze coordination during collaborative multimedia learning." Examines joint attention and CRQA of gaze in dyads.

### Spatial Tasks & Navigation

- **Li et al. (2023)** — "Eye movement patterns in map reading: A systematic review." Reviews fixation, saccade, and pupil metrics specific to map-based tasks.

- **Kiefer et al. (2022)** — "Gaze-Based Cognitive Load Estimation for Spatial Tasks." Uses pupil dilation + fixation duration as cognitive load proxy during map use.

### Foundational Tutorials

- **Richardson & Dale (2005)** — "Looking to understand: The coupling between speakers' and listeners' eye movements and its relationship to discourse comprehension." Foundational for dyadic gaze CRQA. [Link](https://doi.org/10.1207/s15516709cog0000_29)

- **Anderson et al. (2013)** — "A recurrence-based analysis of scanpath dynamics." Applying RQA to gaze data. [Link](https://doi.org/10.3758/s13428-012-0225-0)

- **Krejtz et al. (2014)** — "Gaze transition entropy." AOI-based entropy measures. [Link](https://doi.org/10.1145/2578153.2578176)

---

## 13. Implementation Notes for This Study

### Tracker Configuration
- **Aurora (iMotions)**: Full fixation/saccade event detection built-in. Has `Fixation Duration`, `Saccade Amplitude/Velocity/Direction`, `Gaze Velocity`. Pupil via `ET_PupilLeft/Right`. Blinks from `ET_EyelidOpeningLeft/Right` (threshold <0.2mm).
- **SmartEye Pro 10**: Raw gaze + fixation/saccade indices only. Fixation duration, saccade amplitude/velocity/direction, and gaze velocity must be computed from raw data. Has explicit `Blink` column. Pupil via `LeftPupilDiameter/RightPupilDiameter`.

### Preprocessing Pipeline
1. `preprocess_eye.py` → unified CSV with columns: `t_unix_ms, trial, gaze_x, gaze_y, aoi, pupil_left, pupil_right, head_pitch/yaw/roll, fixation_idx/x/y/duration, saccade_idx/amplitude/peak_velocity/direction, gaze_velocity, blink, role, source`
2. SmartEye gaps (fixation duration, saccade metrics) computed in `gaze_features.py` from raw gaze and event indices
3. Aurora blinks detected from eyelid opening threshold

### Map Coordinate Normalization
Both participants' gaze normalized to common 651x900 map space:
- Director: screen (252,137)-(889,1017) → map (0,0)-(651,900)
- Matcher: screen (267,137)-(904,1017) → map (0,0)-(651,900)

### Sampling Rates
- Aurora: ~60 Hz typical (iMotions)
- SmartEye: ~60 Hz (3-camera setup)
- Resample to common rate (60 Hz) for cross-participant analysis

### Trial Duration
210 seconds per trial, sufficient for all temporal metrics.
