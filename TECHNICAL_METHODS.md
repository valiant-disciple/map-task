# Director–Matcher Map Task: Technical Methods

> Full technical specification of the multimodal dyadic collaboration platform.
> All parameter values are taken directly from source code as of commit `c2dc90b` (2026-03-24).

---

## 1. System Architecture

### 1.1 Components

| Component | Stack | Purpose |
|-----------|-------|---------|
| **map-task-frontend** | React 18 + TypeScript 5.6 + Vite 5 | Experiment UI (Lobby, Director, Matcher, End pages) |
| **backend** | Express + TypeScript + `ws` | HR ingestion, WebSocket relay, audio upload, session sync |
| **replay-dashboard** | Express + Python 3 | Post-session replay, ML/NLP postprocessing pipeline |
| **wear-hr-app** | Kotlin / WearOS (minSdk 30, targetSdk 34) | Smartwatch HR + accelerometer + gyroscope streaming |

### 1.2 Deployment

Three Render.com backend instances (Node 20, free tier):

| Service | Role | URL |
|---------|------|-----|
| watch-hr-session | Session WebSocket (trial sync, clock offset) | `wss://watch-hr-session-u96c.onrender.com` |
| watch-hr-backend | Director HR ingestion + REST | `wss://watch-hr-backend-u96c.onrender.com` |
| watch-hr-backend-matcher | Matcher HR ingestion + REST | `wss://watch-hr-backend-matcher-u96c.onrender.com` |

Frontend served as static SPA via `serve` on Render.

### 1.3 Data Flow

```
WearOS Watch (HR/Accel/Gyro) ──WebSocket──► HR Backend ──HTTP poll──► Frontend
                                                                         │
Director Frontend ◄──────── Session WebSocket ────────► Matcher Frontend
       │                         │                            │
       ▼                         ▼                            ▼
  Director ZIP              Sync Events                 Matcher ZIP
       │                                                     │
       └──────────► replay-dashboard (merge) ◄───────────────┘
                           │
                    postprocess.py
                           │
                    16+ output CSVs
```

---

## 2. Experiment Parameters

### 2.1 Trial Structure

| Parameter | Value |
|-----------|-------|
| Total trials | 8 |
| Warmup trials | 2 (trials 1–2) |
| Data trials | 6 (trials 3–8) |
| Trial duration | 210 seconds |
| Pre-start countdown | 5 seconds |
| Map sets | 2 (Set 1: maps 0–7, Set 2: maps 8–15) |
| Counterbalancing | Data trial map order shuffled; warmups fixed |

### 2.2 Map Assets

- 32 GIF files: `map{0-15}{f,g}.gif`
  - `f` suffix: full map (Director view — shows gold route)
  - `g` suffix: ground map (Matcher view — no route)
- Canvas rendering: 651 × 900 pixels
- Legacy recordings (1024 × 1024) auto-rescaled in postprocessing
- Ground truth routes: `gt_{0-15}.json` (coordinate arrays)

### 2.3 Roles

- **Director**: Views full map (read-only), gives verbal instructions, controls trial start/stop
- **Matcher**: Views ground map (interactive), draws route from instructions, can draw/erase

---

## 3. Frontend — Data Capture

### 3.1 Canvas Drawing (Matcher)

| Parameter | Value |
|-----------|-------|
| Stroke width (draw mode) | 3 px |
| Eraser width | 20 px |
| Stroke color | `#ff0000` (red) |
| Stroke cap/join | Round |
| Erase composite | `destination-out` |
| Input | Mouse + touch (pointer events) |

Each stroke is logged as an `EventRecord`:
```json
{
  "t": 1711234567890,
  "type": "draw_stroke",
  "role": "matcher",
  "payload": {
    "points": [{"x": 324, "y": 450, "t": 1711234567890}, ...],
    "polyline": [...],
    "mode": "draw",
    "mapNumber": 3,
    "trialIndex": 4
  }
}
```

Point coordinates are in canvas pixel space (651 × 900). Each point carries a sub-millisecond local timestamp.

### 3.2 Heart Rate

**WearOS App**:
- Sensors: `TYPE_HEART_RATE`, `TYPE_ACCELEROMETER`, `TYPE_GYROSCOPE`
- Sensor delay: `SENSOR_DELAY_NORMAL` (~200 ms typical)
- WebSocket connection with 3-second auto-reconnect on failure
- JSON payload per event: `{ deviceId, ts, type, values, accuracy }`
- Screen kept always-on (`FLAG_KEEP_SCREEN_ON`)
- Remote start/stop commands from backend

**Frontend HR Polling**:
- Polls backend HTTP endpoint (`GET /api/hr/latest`) every 1500 ms
- Role-specific backends (Director and Matcher have separate HR servers)
- Phases: `baseline` | `trial` | `idle`
- Fallback: simulated HR (72 ± 10 bpm, 1 Hz) after 5 consecutive poll failures

**Baseline Measurement**:
- Duration: 20 seconds
- Update interval: 200 ms
- Computed as mean of all readings during baseline window
- Logged as `baseline_hr` event with `avgBpm`

**Backend Storage**:
- Ring buffer: 1000 records global, 500 per device
- Server-Sent Events (SSE) for real-time streaming
- In-memory only (no persistence across restarts)

### 3.3 Audio Recording

- API: `MediaRecorder` (Web API)
- Codec preference chain:
  1. `audio/webm;codecs=opus`
  2. `audio/webm`
  3. System default (no MIME specified)
- Device selection fallback chain:
  1. `{ deviceId: { exact: selectedId } }`
  2. `{ deviceId: { ideal: selectedId } }` (if exact fails)
  3. `{ audio: true }` (any available mic)
- 100 ms delay after permission grant for driver cleanup (Windows lab PCs)
- Recording starts at trial start, stops at trial end
- Output: one WebM blob per trial per role
- Both roles alert user on recording failure

### 3.4 Surveys

**NASA Raw Task Load Index (NASA-TLX)**:
- Administered after each data trial (trials 3–8)
- 6 dimensions: Mental Demand, Physical Demand, Temporal Demand, Performance, Effort, Frustration
- Scale: 0–100, step size 5
- Default (neutral): 50

**Perceived Shared Mental Model (PSMM)**:
- Administered after each data trial
- 8 items, 1–7 Likert scale, default 4
- Two factors:
  - Task SMM (4 items): route understanding, landmarks, obstacles, position awareness
  - Team SMM (4 items): anticipation, role clarity, communication effectiveness, conflict resolution

**Demographics** (pre-experiment):
- Age (18–100), gender, handedness, native language
- English fluency (1–7), partner familiarity (1–7)
- Prior map task experience (boolean)
- Hearing/vision notes

**Debrief** (post-experiment):
- Overall difficulty (1–7), communication quality (1–7)
- Strategy description, challenges, would-change-approach, general feedback

**Trial Success** (Director only, per data trial):
- Target reached (yes/no), path confidence (1–7), notes

### 3.5 Event Logging

All experiment events are recorded as `EventRecord` objects:
```typescript
{ t: number; type: string; role?: string; payload?: any }
```

Timestamps use `Date.now()` (local machine clock, millisecond precision). Events are auto-saved to `localStorage` every 5 seconds and on page unload for crash recovery.

**Event types logged**:
- `demographics`, `baseline_hr`, `clock_offset`, `sync_flash`
- `draw_stroke`, `action_toggle_mode`
- `trial_final_time` (with `remainSec`, `elapsedSec`, `cause`)
- `tlx_submit`, `psmm_submit`, `trial_success`, `debrief_submit`

---

## 4. Temporal Synchronization

### 4.1 Architecture

Each device logs timestamps on its own local clock. No device clocks are modified during the experiment. Synchronization is achieved through:

1. **NTP-style clock offset measurement** between Director and Matcher laptops
2. **SyncFlash pupil constriction detection** for eye tracker alignment
3. **Post-hoc offset correction** applied in the postprocessing pipeline

All raw data is preserved unmodified; corrections are applied during analysis only.

### 4.2 NTP Clock Offset Protocol

Both Director and Matcher independently measure the clock offset to the other side using the session WebSocket.

**Protocol**:
1. Initiator generates a unique `pingId` (random 8-char alphanumeric)
2. Sends `clock_ping` with `{ pingId, t1: Date.now() }`
3. Responder receives at its own `t2`, immediately sends `clock_pong` with `{ pingId, t1, t2, t3: Date.now() }`
4. Initiator receives at `t4 = Date.now()`
5. Computes: `RTT = (t4 − t1) − (t3 − t2)`, `offset = ((t2 − t1) + (t3 − t4)) / 2`

**Parameters**:

| Parameter | Value |
|-----------|-------|
| Samples per measurement | 5 |
| Interval between pings | 300 ms |
| Timeout | 8000 ms |
| Outlier removal | 20% trimmed mean (top and bottom 20% discarded) |
| Measurement timing | On session join + between every trial |

**Disambiguation**: Each measurement session uses a unique `pingId`. Both sides can measure simultaneously without cross-contaminating samples — a handler only processes `clock_pong` messages whose `pingId` matches its own.

**Timeout handling**: If fewer than 5 pongs arrive within 8 seconds, the measurement resolves with whatever samples were collected. If 0 samples, offset defaults to 0 ms and is corrected at the next inter-trial re-measurement.

**Re-measurement**: The Matcher re-measures the clock offset at every `trial_prepare` event (between trials). This corrects for clock drift over the session duration. Typical laptop clock drift is 1–5 ms/minute; re-measurement every ~4 minutes keeps cumulative drift below 20 ms.

### 4.3 Timer Synchronization

The Director initiates each trial by broadcasting:
```json
{ "event": "start", "payload": { "startAt": <Director_clock_ms + 5000>, "trialIndex": 4, "mapNumber": 7, "durationSec": 210 } }
```

The Matcher adjusts the received `startAt` to its local clock:
```
adjusted_startAt = payload.startAt − clockOffsetRef.current
```

Both sides then count down independently using their local `Date.now()` polled at 50 ms intervals (20 Hz). The trial ends when `Date.now() >= startAt + durationSec × 1000` on each side independently — no network dependency during the trial.

### 4.4 SyncFlash (Eye Tracker Alignment)

A full-screen white flash fires at the exact `startAt` timestamp on both Director and Matcher screens.

**Parameters**:

| Parameter | Value |
|-----------|-------|
| Flash color | `#ffffff` (maximum luminance) |
| Flash duration | 150 ms |
| Z-index | 99999 (covers all UI elements) |
| Tolerance | Flash does not fire if `startAt` passed by > 500 ms |
| Frequency | Once per trial (8 flashes total per session) |

**Screen state transition**: At `startAt`, the screen transitions from normal UI (map + toolbar + sidebar) to full white, then back to normal UI after 150 ms. This luminance spike triggers the pupillary light reflex (constriction onset ~200–500 ms after stimulus).

**Timestamp logging**: The flash callback logs `Date.now()` at the moment the flash fires (not when it was scheduled), producing a `sync_flash` event with `flashTs` in the local machine's clock.

**Eye tracker detection** (in `preprocess_eye.py`):
1. Reads `sync_flash` events from `events.json` → extracts `flashTs`
2. For each flash, searches eye tracker data within ±2000 ms of `flashTs`
3. Computes the steepest negative pupil diameter derivative (mm/ms) across consecutive samples
4. Returns the eye tracker timestamp at constriction onset
5. Offset = `eye_tracker_onset_time − flashTs`
6. Takes median of all per-trial offsets as the final alignment correction

Minimum 10 pupil samples required within the search window for valid detection.

### 4.5 Resilience

| Scenario | Protection |
|----------|-----------|
| Peer not connected during offset measurement | 8 s timeout → defaults to 0, corrected at next trial boundary |
| Clock drift over 30-min session | Re-measured between every trial via `trial_prepare` |
| WebSocket drops mid-trial | Both sides auto-end independently on local timeout |
| WebSocket drops between trials | Auto-reconnect (1–5 s backoff), message queue flushes on reconnect, `sync_request` on rejoin |
| Both sides measure offset simultaneously | `pingId` prevents cross-contamination |
| Partial measurement (< 5 samples) | Resolves with trimmed mean of available samples |
| Browser tab throttling | Timestamps use `Date.now()` (unaffected); only display refresh rate degrades |

### 4.6 WebSocket Configuration

| Parameter | Value |
|-----------|-------|
| Keep-alive ping interval | 25 seconds |
| Reconnect backoff | 1 s → 2 s → 4 s → 5 s (cap) |
| Message queue | Queued while disconnected, flushed on reconnect |

---

## 5. Data Export

### 5.1 ZIP Structure (Per Role, Independent Download)

Each participant downloads their own ZIP. No merging required at capture time.

```
map_task_{role}_{sessionId}.zip
├── session/
│   ├── session.json          # Session config, trial summaries
│   ├── events.json           # All events for this role
│   ├── hr_baseline.json      # Baseline HR measurement
│   └── sync.csv              # Clock offsets + flash timestamps
├── trials/
│   ├── T01/
│   │   ├── events.json       # Trial-scoped events
│   │   ├── strokes.json      # Matcher strokes (cleaned)
│   │   ├── cursor.json       # Cursor position log
│   │   ├── tlx_{role}.json   # NASA-TLX responses
│   │   ├── psmm_{role}.json  # PSMM responses
│   │   ├── final_image.png   # Rendered matcher drawing
│   │   ├── audio/
│   │   │   └── {role}_T{n}.webm
│   │   └── hr/
│   │       └── hr_{role}.csv
│   ├── T02/ ...
│   └── T08/ ...
```

**sync.csv columns**: `type, role, trial, t_unix_ms, offsetMs, rttMs, samples, peerRole, flashTs`

**HR CSV columns**: `timestamp_unix_ms, timestamp_iso, bpm, phase`

### 5.2 Stroke Cleaning

Before export, strokes are filtered to remove noise:
- Minimum 2 points per stroke
- Minimum path length > 6 px
- Minimum endpoint displacement > 1 px (Euclidean)
- Valid modes only: `draw` or `erase`

---

## 6. Postprocessing Pipeline

### 6.1 Overview

The replay dashboard (`replay-dashboard/`) accepts both Director and Matcher ZIPs, merges them client-side, and runs `postprocess.py` server-side. The pipeline produces 16+ CSVs covering drawing accuracy, HRV, physiological synchrony, speech, gaze, and behavioral metrics.

### 6.2 Clock Offset Alignment

`extract_clock_offset()` reads `clock_offset` events from the merged ZIP's `session/events.json`:

- Matcher event (`role='matcher', peerRole='director'`): `offsetMs` = Director − Matcher → used directly
- Director event (`role='director', peerRole='matcher'`): `offsetMs` = Matcher − Director → negated
- If both measurements present: averaged
- Convention: `matcher_offset = Director_clock − Matcher_clock` (positive = Director ahead)

**Application**: All Matcher timestamps are shifted by `+ matcher_offset`:
- HR timestamps: `apply_offset_to_hr()` mutates in-place
- Stroke timestamps: `_offset_t()` applied to all stroke and point-level timestamps
- Gaze timestamps: shifted after eye-tracker preprocessing
- Manifest timestamps: offset-corrected for trial boundaries

### 6.3 HR Cross-Dyad Alignment

After offset correction, both HR series are on the Director's reference clock but may have different sampling rates and missing samples. Before cross-dyad analysis:

`interpolate_hr_pair(hr_m, hr_d, sample_interval_ms=1000)`:
1. Extracts valid (timestamp, bpm) pairs from both series
2. Computes overlapping time range
3. Creates common 1 Hz grid (1000 ms intervals)
4. Linearly interpolates both series onto the grid via `numpy.interp`
5. Returns equal-length BPM arrays for cross-dyad analysis

### 6.4 Drawing Metrics

Per-trial comparison of Matcher's drawn path against ground truth:

| Metric | Method |
|--------|--------|
| IoU | Intersection over Union of binary masks |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | Harmonic mean of precision and recall |
| Dice | 2 × |A ∩ B| / (|A| + |B|) |
| SSIM | Structural Similarity Index |
| Hausdorff | Maximum of minimum distances |
| Chamfer | Mean of minimum distances |
| Boundary F1 | F1 at 2-pixel tolerance on contour boundaries |
| Coverage (GT) | Fraction of GT mask that is non-zero |
| Coverage (Pred) | Fraction of predicted mask that is non-zero |

Time-series correctness: metrics recomputed incrementally at every 5th stroke to track accuracy progression.

### 6.5 Heart Rate Variability

**Tier 1 — Time Domain**:
- `bpm_mean`, `bpm_std`, `bpm_min`, `bpm_max`, `hr_range`
- IBI conversion: `IBI_ms = 60000 / BPM`
- `mean_rr_ms`, `sdnn_ms` (SD of NN intervals)
- `rmssd_ms` (root mean square of successive differences)
- `ln_rmssd` (natural log of RMSSD)
- `nn50`, `pnn50` (count/% of successive intervals differing > 50 ms)
- `sdsd_ms` (SD of successive differences)

**Tier 2 — Nonlinear**:
- Sample Entropy: `m=2`, `r=0.2 × SD`
- DFA α1: Detrended Fluctuation Analysis, box sizes 4–16
- Poincaré: `SD1`, `SD2`, `SD1/SD2` ratio

**Tier 3 — Frequency Domain**:
- IBI resampled to 4 Hz via cubic spline interpolation
- Welch PSD with 60 s window
- VLF: 0.003–0.04 Hz
- LF: 0.04–0.15 Hz
- HF: 0.15–0.40 Hz
- Derived: `lf_hf_ratio`, `lf_nu`, `hf_nu`, `total_power_ms2`

### 6.6 Physiological Synchrony (Cross-Dyad HR)

All analyses operate on the time-aligned, interpolated 1 Hz BPM series.

**Auto-RQA** (per individual):
- Embedding dimension: 2, time delay: 1
- Threshold: `max(0.1 × SD, 0.01)`
- Metrics: `rqa_rr`, `rqa_det`, `rqa_mean_diag`, `rqa_max_diag`, `rqa_div`, `rqa_entr_diag`, `rqa_lam`, `rqa_tt`, `rqa_entr_vert`, `rqa_max_vert`

**Cross-RQA (CRQA)**:
- Same embedding parameters
- Threshold: `max(0.1 × SD_combined, 0.01)`
- Theiler corrector: 0
- Full metric set prefixed `crqa_*`

**Multidimensional RQA (MdRQA)**:
- Joint 2D phase space: `[matcher_HR, director_HR]`
- Z-score normalization per channel
- Threshold: 5th percentile of Euclidean distance matrix
- Full metric set prefixed `mdrqa_*`

**Diagonal Cross-Recurrence Profile (DCRP)**:
- Lag range: −20 to +20 samples
- Metrics: `dcrp_peak_lag`, `dcrp_peak_rr`, `dcrp_width`, `dcrp_los_rr`

**Windowed Cross-Correlation**:
- Window: 30 s, step: 10 s, sample interval: 1.5 s
- Pearson r per window
- Metrics: `wcc_mean_r`, `wcc_max_r`, `wcc_min_r`, `wcc_std_r`, `wcc_n_windows`, `wcc_pct_positive`

**Transfer Entropy** (symbolic):
- Binary encoding: diff(HR) → increase/decrease
- Bidirectional: matcher→director, director→matcher
- `te_asymmetry = TE(director→matcher) − TE(matcher→director)`

**Windowed CRQA**:
- Window: 60 s, step: 30 s, sample interval: 1.5 s
- Tracks RR and DET evolution over trial
- Reports mean, SD, and linear trend

**Surrogate Baseline**:
- 20 time-shifted surrogates (random circular shift within 25–75% of series length)
- Z-score real vs. surrogate distribution
- Reports `surr_rr_z`, `surr_det_z` for significance testing

### 6.7 Audio & Speech

**Prosody** (librosa):
- F0 via YIN: fmin=50 Hz, fmax=500 Hz
- Features: `duration_sec`, `rms_mean`, `rms_std`, `zcr_mean`, `f0_mean`, `f0_median`, `f0_coverage`

**ASR** (optional, requires API key):
- Smallest Pulse API: text + confidence
- Whisper (OpenAI): word-level timestamps, segments

**Full Speech Pipeline** (optional, requires openai key):
- Per-role: Whisper ASR + Parselmouth prosody (F0, intensity, jitter, shimmer, HNR, formants F1–F4) + OpenSMILE eGeMAPSv02 (40+ acoustic functionals)
- Dyad-level: turn-taking, overlap, response latency, acoustic similarity

**LLM Dialogue Evaluation** (optional, GPT-4.1-mini):
- 10 dialogue act categories: INSTRUCT, DESCRIBE, CHECK, QUERY, CLARIFY, ACKNOWLEDGE, REPAIR, META, FILLER, OTHER
- Communication quality (8 dimensions, 1–7 Likert)
- Linguistic convergence (5 dimensions, 1–7 Likert)

### 6.8 Eye Tracker Preprocessing

**Supported formats**:
- Aurora (iMotions CSV): timestamps relative to recording start, converted via Unix header
- SmartEye Pro 10 (.log TSV): Windows FILETIME (100 ns intervals since 1601-01-01), converted via `RTC / 10000 − 11644473600000`

**AOI Definitions** (pixel coordinates):

| AOI | Director | Matcher |
|-----|----------|---------|
| Map | x: 252–889, y: 137–1017 | x: 267–904, y: 137–1017 |
| Timer | x: 613–735, y: 8–65 | x: 613–735, y: 8–65 |
| Toolbar | x: 236–1018, y: 0–74 | x: 236–1365, y: 0–74 |

**Flash-based alignment**:
1. Search window: ±2000 ms around `flashTs`
2. Detection: steepest negative pupil diameter derivative (mm/ms)
3. Minimum 10 samples required
4. Final offset: median of all per-trial flash offsets

**Output columns** (24):
`t_unix_ms`, `t_iso`, `trial`, `gaze_x`, `gaze_y`, `aoi`, `pupil_left`, `pupil_right`, `head_pitch`, `head_yaw`, `head_roll`, `fixation_idx`, `fixation_x`, `fixation_y`, `fixation_duration`, `saccade_idx`, `saccade_amplitude`, `saccade_peak_velocity`, `saccade_direction`, `gaze_velocity`, `blink`, `eyelid_left`, `eyelid_right`, `role`, `source`

### 6.9 Gaze Features (100+ metrics)

**Per-individual**:
- Fixation: count, rate, duration (mean/median/SD/max), spatial dispersion
- Saccade: count, amplitude (mean/SD), peak velocity, direction distribution, regressive rate
- Pupil: mean diameter, SD, TEPR, ICA, low-frequency power
- Blink: rate, duration, inhibition
- Scanpath: length, convex hull, entropy (SGE), transition entropy (GTE), nearest-neighbor index, RQA
- AOI: dwell time per AOI, fixation count, transition matrix, coverage, revisits

**Dyadic coupling**:
- CRQA of gaze: RR, DET, L, Lmax, entropy, LAM
- Cross-correlation: lag at peak, peak r
- Joint AOI fixation proportion
- Gaze convergence index (Euclidean distance)
- Leader-follower asymmetry

### 6.10 Drawing Behavior Features

**Temporal**: stroke count, active drawing time, duty cycle, inter-stroke intervals, hesitations, drawing pace trend

**Kinematic**: stroke lengths, displacements, straightness, point density, speed, acceleration, curvature

**Behavioral**: erase ratio, backtracking, speed bursts

**Spatial**: bounding box area, coverage growth, velocity near boundaries

**Complexity**: fractal dimension (box-counting), visual complexity

### 6.11 Knowledge Graph (Optional, GPT-4.1)

- Vision: extract landmarks from map images (cached)
- Text: extract spatial relations from dialogue (source → target, relation type)
- Graph construction: NetworkX directed graph
- Comparison to ground truth graph

---

## 7. Output CSVs

| CSV | Granularity | Key Columns |
|-----|-------------|-------------|
| `metrics.csv` | Per trial | IoU, F1, SSIM, Hausdorff, Chamfer, boundary metrics |
| `time_series_metrics.csv` | Per stroke | Incremental accuracy at each 5th stroke |
| `strokes.csv` | Per point | sessionId, trial, x, y, t_unix_ms, mode |
| `hr_matcher.csv` | Per sample | t_unix_ms (offset-corrected), bpm, phase |
| `hr_director.csv` | Per sample | t_unix_ms, bpm, phase |
| `hr_stats.csv` | Per trial × role + cross | All HRV metrics + cross-dyad synchrony |
| `prosody.csv` | Per audio file | duration, RMS, ZCR, F0 |
| `speech.csv` | Per audio file | ASR transcript, confidence |
| `speech_features.csv` | Per trial × role | Whisper + Parselmouth + OpenSMILE |
| `pair_speech.csv` | Per trial | Turn-taking, overlap, acoustic similarity |
| `llm_eval.csv` | Per trial | Dialogue acts, quality, convergence |
| `knowledge_graph.csv` | Per trial | Landmarks, relations, graph similarity |
| `gaze_features.csv` | Per trial × role | 100+ fixation/saccade/pupil/AOI metrics |
| `gaze_pair.csv` | Per trial | Joint attention, gaze leading, multimodal |
| `drawing_features.csv` | Per trial | Temporal, kinematic, behavioral, spatial |
| `manifest.csv` | Per trial | Metadata, TLX, PSMM, clock_offset_ms, ref_clock |

---

## 8. Temporal Alignment Summary

### 8.1 Reference Clock Convention

All postprocessed timestamps are normalized to the **Director's laptop clock**.

### 8.2 Alignment Chain Per Modality

| Modality | Source Clock | Alignment Method | Precision |
|----------|-------------|------------------|-----------|
| Director HR | Director laptop | Reference (no correction) | Native |
| Director strokes/events | Director laptop | Reference (no correction) | Native |
| Matcher HR | Matcher laptop | `+ matcher_offset` (NTP, re-measured per trial) | < 20 ms |
| Matcher strokes/events | Matcher laptop | `+ matcher_offset` | < 20 ms |
| Director eye tracker | Eye tracker HW | SyncFlash pupil detection → Director laptop | < 16 ms (1 frame @ 60 Hz) |
| Matcher eye tracker | Eye tracker HW | SyncFlash → Matcher laptop → `+ matcher_offset` | < 36 ms |
| Audio | Local clock | Relative features only (pitch, energy, duration) | N/A |
| Surveys | N/A | No timestamps (ratings only) | N/A |

### 8.3 Worst-Case Error Budget

| Source | Contribution |
|--------|-------------|
| NTP measurement noise | ± 5 ms (trimmed mean of 5 samples) |
| Clock drift between re-measurements | < 20 ms (re-measured every ~4 min) |
| `setTimeout` jitter (SyncFlash) | 1–4 ms |
| Eye tracker frame aliasing (60 Hz) | ± 8 ms |
| **Total worst case** | **< 37 ms** |

For HR data sampled at 1.5 s intervals (667 ms per sample), this is < 6% of one sample. For eye tracking at 60 Hz (16.7 ms per frame), this is within 2–3 frames.

---

## 9. Microphone Access Hardening

### 9.1 Device Enumeration

Permission-aware enumeration: skips acquire-release if permission already granted (checks for populated `label` fields). When permission is needed, acquires a temporary stream, releases it, waits 100 ms for driver cleanup, then enumerates.

### 9.2 Recording Start Fallback

Three-tier fallback chain:
1. `{ deviceId: { exact: selectedId } }` — preferred device
2. `{ deviceId: { ideal: selectedId } }` — best-effort preferred
3. `{ audio: true }` — any available microphone

### 9.3 Failure Handling

Both Director and Matcher alert the user if recording fails at trial start. The trial proceeds (audio loss is logged, not blocking), but the experimenter is notified immediately.

### 9.4 Volume Meter

Uses `{ deviceId: { ideal: selectedId } }` (not `exact`) for the real-time volume meter in the mic check widget, avoiding failures on lab PCs with strict audio drivers.

---

## 10. Dependencies

### Frontend
- React 18.3, React Router 6, TypeScript 5.6, Vite 5
- jszip (ZIP export), react-speech-recognition (mic check)

### Backend
- Express, ws (WebSocket), Zod (validation), multer, cors
- Node 20 (Docker), ffmpeg-static

### WearOS
- OkHttp3 4.12.0, Kotlin Coroutines Android 1.9.0
- AndroidX Core 1.13.1, Activity 1.9.2, Lifecycle 2.8.6

### Postprocessing (Python)
- numpy, scipy, Pillow (core)
- librosa, soundfile (audio)
- pyrqa (recurrence quantification analysis)
- requests (ASR API)
- Optional: openai, parselmouth, opensmile, networkx
