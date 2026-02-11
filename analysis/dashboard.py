#!/usr/bin/env python3
"""
Map Task Analysis Dashboard
============================
Interactive Streamlit dashboard for viewing processed session data.

Run:
    streamlit run dashboard.py
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─── Page Config ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Map Task Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Helpers ─────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def find_sessions() -> list:
    """Find all processed sessions in the data directory."""
    if not os.path.isdir(DATA_DIR):
        return []
    sessions = []
    for name in sorted(os.listdir(DATA_DIR)):
        session_dir = os.path.join(DATA_DIR, name)
        summary_path = os.path.join(session_dir, "processing_summary.json")
        if os.path.isdir(session_dir) and os.path.exists(summary_path):
            sessions.append(name)
    return sessions


def load_summary(session_id: str) -> dict:
    """Load processing_summary.json for a session."""
    path = os.path.join(DATA_DIR, session_id, "processing_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_json(path: str) -> dict | list | None:
    """Safely load a JSON file."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_trial_hr_processed(session_id: str, trial_name: str) -> dict | None:
    """Load the processed HR JSON for a trial."""
    path = os.path.join(DATA_DIR, session_id, "trials", trial_name, "hr", "hr_processed.json")
    return load_json(path)


def load_trial_speech(session_id: str, trial_name: str, role: str) -> dict | None:
    """Load speech analysis JSON for a specific role in a trial."""
    trial_dir = os.path.join(DATA_DIR, session_id, "trials", trial_name, "audio")
    if not os.path.isdir(trial_dir):
        return None
    for f in os.listdir(trial_dir):
        if f.endswith("_analysis.json") and role in f.lower():
            return load_json(os.path.join(trial_dir, f))
    return None


def load_trial_path(session_id: str, trial_name: str) -> dict | None:
    """Load path analysis JSON for a trial."""
    path = os.path.join(DATA_DIR, session_id, "trials", trial_name, "path_analysis.json")
    return load_json(path)


def load_trial_surveys(session_id: str, trial_name: str) -> dict | None:
    """Load surveys processed JSON for a trial."""
    path = os.path.join(DATA_DIR, session_id, "trials", trial_name, "surveys_processed.json")
    return load_json(path)


# ─── Sidebar ─────────────────────────────────────────────────────────

st.sidebar.title("🗺️ Map Task Dashboard")
st.sidebar.markdown("---")

sessions = find_sessions()
if not sessions:
    st.sidebar.warning("No processed sessions found.")
    st.sidebar.info(f"Run `python process_session.py <zip>` first.\nData directory: `{DATA_DIR}`")
    st.title("🗺️ Map Task Analysis Dashboard")
    st.info("No processed sessions found. Process a session ZIP first:\n\n```bash\ncd analysis\npython process_session.py path/to/session.zip\n```")
    st.stop()

selected_session = st.sidebar.selectbox("📂 Session", sessions)
summary = load_summary(selected_session)

if not summary:
    st.error(f"Could not load summary for session: {selected_session}")
    st.stop()

trial_names = sorted(summary.get("trials", {}).keys())
view = st.sidebar.radio(
    "View",
    ["📊 Session Overview", "🔍 Trial Detail", "❤️ HR Analysis", "🎤 Speech Analysis", "🗺️ Path Analysis", "📋 Surveys", "📈 Cross-Trial Trends"],
)

# ─── Session Overview ────────────────────────────────────────────────

if view == "📊 Session Overview":
    st.title("📊 Session Overview")

    # Header metrics
    config = summary.get("config", {})
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Session ID", summary.get("session_id", "?"))
    col2.metric("Map Set", config.get("mapSet", "?"))
    col3.metric("Data Trials", len(trial_names))
    col4.metric("Duration/Trial", f"{config.get('durationSec', '?')}s")
    col5.metric("Processed", summary.get("processed_at", "?")[:10])

    st.markdown("---")

    # Demographics
    demographics = summary.get("demographics", {})
    if demographics:
        st.subheader("👥 Participants")
        demo_cols = st.columns(2)
        for i, (role, data) in enumerate(demographics.items()):
            with demo_cols[i % 2]:
                st.markdown(f"**{role.title()}**")
                if isinstance(data, dict):
                    for k, v in data.items():
                        st.text(f"  {k}: {v}")

    # Trial overview table
    st.subheader("📋 Trial Summary")
    rows = []
    for tname, tdata in sorted(summary.get("trials", {}).items()):
        row = {"Trial": tname}

        # Surveys
        surveys = tdata.get("surveys", {})
        row["TLX Dir"] = surveys.get("tlx_director", {}).get("overall_workload", "-")
        row["TLX Mat"] = surveys.get("tlx_matcher", {}).get("overall_workload", "-")
        row["PSMM Dir"] = surveys.get("psmm_director", {}).get("overall_smm_mean", "-")
        row["PSMM Mat"] = surveys.get("psmm_matcher", {}).get("overall_smm_mean", "-")

        # Trial success
        ts = surveys.get("trial_success", {})
        row["Target Reached"] = "✅" if ts.get("target_reached") else ("❌" if ts.get("target_reached") is False else "-")
        row["Path Confidence"] = ts.get("path_confidence", "-")

        # HR
        hr = tdata.get("hr", {})
        row["HR Dir (mean)"] = hr.get("director", {}).get("stats", {}).get("bpm_mean", "-")
        row["HR Mat (mean)"] = hr.get("matcher", {}).get("stats", {}).get("bpm_mean", "-")
        sync = hr.get("synchrony", {})
        row["HR Sync (r)"] = sync.get("pearson_r", "-")

        # Path
        path = tdata.get("path", {})
        row["Path Length"] = path.get("path_length_px", "-")
        row["Strokes"] = path.get("num_strokes", "-")

        # Speech
        speech = tdata.get("speech", {})
        for role in ["director", "matcher"]:
            sp = speech.get(role, {})
            prosody = sp.get("prosody", {})
            rate = prosody.get("rate", {})
            row[f"Speech % ({role[:3]})"] = rate.get("speech_ratio", "-")

        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Debrief
    debrief = summary.get("debrief", {})
    if debrief:
        st.subheader("💬 Debrief")
        for k, v in debrief.items():
            if v:
                st.markdown(f"**{k.title()}:** {v}")


# ─── Trial Detail ────────────────────────────────────────────────────

elif view == "🔍 Trial Detail":
    st.title("🔍 Trial Detail")

    if not trial_names:
        st.warning("No trials found.")
        st.stop()

    selected_trial = st.sidebar.selectbox("Trial", trial_names)
    trial_data = summary.get("trials", {}).get(selected_trial, {})

    st.header(f"Trial: {selected_trial}")

    # Quick stats
    cols = st.columns(4)

    surveys = trial_data.get("surveys", {})
    ts = surveys.get("trial_success", {})
    cols[0].metric("Target Reached", "✅ Yes" if ts.get("target_reached") else "❌ No" if ts.get("target_reached") is False else "N/A")
    cols[1].metric("Path Confidence", f"{ts.get('path_confidence', '?')}/7")

    hr = trial_data.get("hr", {})
    sync_r = hr.get("synchrony", {}).get("pearson_r")
    cols[2].metric("HR Synchrony (r)", f"{sync_r:.3f}" if sync_r is not None else "N/A")

    path = trial_data.get("path", {})
    cols[3].metric("Path Length", f"{path.get('path_length_px', '?')} px")

    st.markdown("---")

    # Expandable sections
    with st.expander("📊 Survey Data", expanded=True):
        scol1, scol2 = st.columns(2)
        with scol1:
            st.markdown("**NASA-TLX**")
            for role in ["director", "matcher"]:
                tlx = surveys.get(f"tlx_{role}", {})
                if "error" not in tlx:
                    st.markdown(f"*{role.title()}* — Overall: **{tlx.get('overall_workload', '?')}**")
                    tlx_df = pd.DataFrame([{k: v for k, v in tlx.items() if k != "overall_workload"}])
                    st.dataframe(tlx_df, hide_index=True)
        with scol2:
            st.markdown("**PSMM**")
            for role in ["director", "matcher"]:
                psmm = surveys.get(f"psmm_{role}", {})
                if "error" not in psmm:
                    st.markdown(f"*{role.title()}* — Task SMM: **{psmm.get('task_smm_mean', '?')}** | Team SMM: **{psmm.get('team_smm_mean', '?')}**")

    with st.expander("❤️ HR Data", expanded=False):
        hr_data = load_trial_hr_processed(selected_session, selected_trial)
        if hr_data:
            sync = hr_data.get("synchrony", {})
            if "t_sec" in sync:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sync["t_sec"], y=sync["director_bpm"], name="Director", line=dict(color="#e74c3c")))
                fig.add_trace(go.Scatter(x=sync["t_sec"], y=sync["matcher_bpm"], name="Matcher", line=dict(color="#3498db")))
                fig.update_layout(title="Aligned HR (baseline-corrected)", xaxis_title="Time (s)", yaxis_title="BPM (corrected)", height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No HR data for this trial.")

    with st.expander("🎤 Speech Data", expanded=False):
        speech = trial_data.get("speech", {})
        for role in ["director", "matcher"]:
            sp = speech.get(role, {})
            if "error" in sp:
                st.warning(f"{role.title()}: {sp['error']}")
                continue
            transcript = sp.get("transcript", {})
            prosody = sp.get("prosody", {})
            st.markdown(f"**{role.title()}**")
            if transcript.get("text"):
                st.text_area(f"Transcript ({role})", transcript["text"], height=100, disabled=True)
            rate = prosody.get("rate", {})
            if rate:
                st.text(f"  Duration: {rate.get('total_duration_sec', '?')}s | "
                        f"Speech: {rate.get('speech_duration_sec', '?')}s ({rate.get('speech_ratio', '?')}) | "
                        f"Pauses: {rate.get('num_pauses', '?')}")

    with st.expander("🗺️ Path Data", expanded=False):
        path_data = load_trial_path(selected_session, selected_trial)
        if path_data and "error" not in path_data:
            pcols = st.columns(4)
            pcols[0].metric("Strokes", path_data.get("num_strokes", 0))
            pcols[1].metric("Points", path_data.get("total_points", 0))
            pcols[2].metric("Path Length", f"{path_data.get('path_length_px', 0)} px")
            if "frechet_distance_px" in path_data:
                pcols[3].metric("Fréchet Dist", f"{path_data['frechet_distance_px']} px")

            # Load strokes for visualization
            strokes_path = os.path.join(DATA_DIR, selected_session, "trials", selected_trial, "strokes.json")
            strokes = load_json(strokes_path)
            if strokes:
                fig = go.Figure()
                for i, stroke in enumerate(strokes):
                    if stroke.get("mode") == "erase":
                        continue
                    polyline = stroke.get("polyline", [])
                    if not polyline:
                        continue
                    xs = [p.get("x", p[0]) if isinstance(p, dict) else p[0] for p in polyline]
                    ys = [p.get("y", p[1]) if isinstance(p, dict) else p[1] for p in polyline]
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Stroke {i+1}",
                                            line=dict(width=2), showlegend=False))
                fig.update_layout(title="Matcher Drawn Path", height=500,
                                  yaxis=dict(autorange="reversed"), xaxis=dict(scaleanchor="y"))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No path data for this trial.")


# ─── HR Analysis ─────────────────────────────────────────────────────

elif view == "❤️ HR Analysis":
    st.title("❤️ Heart Rate Analysis")

    if not trial_names:
        st.warning("No trials found.")
        st.stop()

    selected_trial = st.sidebar.selectbox("Trial", trial_names)
    hr_data = load_trial_hr_processed(selected_session, selected_trial)

    if not hr_data:
        st.info("No processed HR data available. Run the processing pipeline first.")
        st.stop()

    # Per-role stats
    st.subheader("Individual HR Stats")
    cols = st.columns(2)
    for i, role in enumerate(["director", "matcher"]):
        with cols[i]:
            rd = hr_data.get(role, {})
            stats = rd.get("stats", {})
            if stats:
                st.markdown(f"### {role.title()}")
                mcols = st.columns(3)
                mcols[0].metric("Mean BPM", stats.get("bpm_mean", "?"))
                mcols[1].metric("Std BPM", stats.get("bpm_std", "?"))
                mcols[2].metric("Baseline", stats.get("baseline_mean", "?"))

                # Individual time series
                ts = rd.get("timeseries", [])
                if ts:
                    df = pd.DataFrame(ts)
                    fig = px.line(df, x="t_sec", y="bpm", title=f"{role.title()} HR")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

    # Synchronized view
    st.subheader("Synchronized HR")
    sync = hr_data.get("synchrony", {})
    if "error" in sync:
        st.warning(sync["error"])
    elif "t_sec" in sync:
        cols = st.columns(3)
        cols[0].metric("Pearson r", sync.get("pearson_r", "?"))
        cols[1].metric("Duration", f"{sync.get('duration_sec', '?')}s")
        cols[2].metric("Samples", sync.get("samples", "?"))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sync["t_sec"], y=sync["director_bpm"],
                                 name="Director", line=dict(color="#e74c3c", width=2)))
        fig.add_trace(go.Scatter(x=sync["t_sec"], y=sync["matcher_bpm"],
                                 name="Matcher", line=dict(color="#3498db", width=2)))
        fig.update_layout(title="Aligned HR Time Series (Baseline-Corrected)",
                          xaxis_title="Time (s)", yaxis_title="BPM (corrected)",
                          height=450, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    # Windowed correlation
    windowed = hr_data.get("windowed_correlation", [])
    if windowed:
        st.subheader("Windowed Correlation (30s sliding window)")
        wdf = pd.DataFrame(windowed)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wdf["t_sec"], y=wdf["r"], mode="lines+markers",
                                 name="Pearson r", line=dict(color="#2ecc71", width=2)))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=0.3, line_dash="dot", line_color="orange", annotation_text="r=0.3")
        fig.add_hline(y=-0.3, line_dash="dot", line_color="orange")
        fig.update_layout(title="Sliding Window HR Correlation",
                          xaxis_title="Time (s)", yaxis_title="Pearson r",
                          height=400, yaxis=dict(range=[-1, 1]))
        st.plotly_chart(fig, use_container_width=True)


# ─── Speech Analysis ─────────────────────────────────────────────────

elif view == "🎤 Speech Analysis":
    st.title("🎤 Speech Analysis")

    if not trial_names:
        st.warning("No trials found.")
        st.stop()

    selected_trial = st.sidebar.selectbox("Trial", trial_names)
    trial_data = summary.get("trials", {}).get(selected_trial, {})
    speech = trial_data.get("speech", {})

    for role in ["director", "matcher"]:
        st.subheader(f"{role.title()}")
        sp = speech.get(role, {})

        if "error" in sp:
            st.warning(sp["error"])
            continue

        # Transcript
        transcript = sp.get("transcript", {})
        if transcript.get("text"):
            st.markdown("**Transcript:**")
            st.text_area(f"transcript_{role}", transcript["text"], height=120, disabled=True, label_visibility="collapsed")

            # Word-level timeline
            segments = transcript.get("segments", [])
            if segments:
                with st.expander("📜 Segments with timestamps"):
                    for seg in segments:
                        st.text(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")

        # Prosody
        prosody = sp.get("prosody", {})
        if prosody:
            pcols = st.columns(4)

            pitch = prosody.get("pitch", {}).get("stats", {})
            pcols[0].metric("Pitch Mean", f"{pitch.get('mean_hz', 0):.0f} Hz")
            pcols[1].metric("Pitch Range", f"{pitch.get('range_hz', 0):.0f} Hz")

            rate = prosody.get("rate", {})
            pcols[2].metric("Speech Ratio", f"{rate.get('speech_ratio', 0):.1%}")
            pcols[3].metric("Pauses", rate.get("num_pauses", 0))

            # Pitch time series
            pitch_ts = prosody.get("pitch", {}).get("timeseries", [])
            if pitch_ts:
                pdf = pd.DataFrame(pitch_ts)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    subplot_titles=["Pitch (F0)", "Energy (RMS)"])

                fig.add_trace(go.Scatter(x=pdf["t"], y=pdf["hz"], mode="lines",
                                         name="Pitch", line=dict(color="#9b59b6")), row=1, col=1)

                energy_ts = prosody.get("energy", {}).get("timeseries", [])
                if energy_ts:
                    edf = pd.DataFrame(energy_ts)
                    fig.add_trace(go.Scatter(x=edf["t"], y=edf["rms"], mode="lines",
                                             name="Energy", line=dict(color="#e67e22")), row=2, col=1)

                fig.update_layout(height=500, showlegend=False)
                fig.update_xaxes(title_text="Time (s)", row=2, col=1)
                fig.update_yaxes(title_text="Hz", row=1, col=1)
                fig.update_yaxes(title_text="RMS", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)

            # Pause visualization
            pauses = prosody.get("pauses", [])
            if pauses:
                with st.expander(f"⏸️ Pauses ({len(pauses)})"):
                    pause_df = pd.DataFrame(pauses)
                    st.dataframe(pause_df, use_container_width=True, hide_index=True)

        st.markdown("---")


# ─── Path Analysis ───────────────────────────────────────────────────

elif view == "🗺️ Path Analysis":
    st.title("🗺️ Path Analysis")

    if not trial_names:
        st.warning("No trials found.")
        st.stop()

    selected_trial = st.sidebar.selectbox("Trial", trial_names)
    path_data = load_trial_path(selected_session, selected_trial)

    if not path_data or "error" in (path_data or {}):
        st.info("No path data for this trial.")
        st.stop()

    cols = st.columns(5)
    cols[0].metric("Strokes", path_data.get("num_strokes", 0))
    cols[1].metric("Total Points", path_data.get("total_points", 0))
    cols[2].metric("Simplified Points", path_data.get("simplified_points", 0))
    cols[3].metric("Path Length", f"{path_data.get('path_length_px', 0):.0f} px")
    if "frechet_distance_px" in path_data:
        cols[4].metric("Fréchet Distance", f"{path_data['frechet_distance_px']:.0f} px")

    if "dtw_distance_px" in path_data:
        st.metric("DTW Distance", f"{path_data['dtw_distance_px']:.0f} px (normalized: {path_data.get('dtw_normalized_px', '?')})")

    # Path visualization
    strokes_path = os.path.join(DATA_DIR, selected_session, "trials", selected_trial, "strokes.json")
    strokes = load_json(strokes_path)
    if strokes:
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, stroke in enumerate(strokes):
            if stroke.get("mode") == "erase":
                continue
            polyline = stroke.get("polyline", [])
            if not polyline:
                continue
            xs = [p.get("x", p[0]) if isinstance(p, dict) else p[0] for p in polyline]
            ys = [p.get("y", p[1]) if isinstance(p, dict) else p[1] for p in polyline]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                name=f"Stroke {i+1} ({stroke.get('role', '?')})",
                line=dict(width=2, color=colors[i % len(colors)]),
            ))

        fig.update_layout(
            title="Matcher Drawn Path",
            height=600,
            yaxis=dict(autorange="reversed", scaleanchor="x"),
            xaxis=dict(constrain="domain"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Final image
    img_path = os.path.join(DATA_DIR, selected_session, "trials", selected_trial, "final_image.png")
    if os.path.exists(img_path):
        st.subheader("Final Map Image")
        st.image(img_path, use_container_width=True)


# ─── Surveys ─────────────────────────────────────────────────────────

elif view == "📋 Surveys":
    st.title("📋 Survey Results")

    if not trial_names:
        st.warning("No trials found.")
        st.stop()

    selected_trial = st.sidebar.selectbox("Trial", trial_names)
    surveys = load_trial_surveys(selected_session, selected_trial)

    if not surveys:
        st.info("No survey data for this trial.")
        st.stop()

    # Trial Success
    st.subheader("🎯 Trial Success")
    ts = surveys.get("trial_success", {})
    if ts.get("reported"):
        cols = st.columns(3)
        cols[0].metric("Target Reached", "✅ Yes" if ts.get("target_reached") else "❌ No")
        cols[1].metric("Path Confidence", f"{ts.get('path_confidence', '?')}/7")
        if ts.get("note"):
            cols[2].info(f"Note: {ts['note']}")
    else:
        st.info("No trial success data reported.")

    st.markdown("---")

    # TLX
    st.subheader("📊 NASA-TLX")
    tlx_cols = st.columns(2)
    for i, role in enumerate(["director", "matcher"]):
        with tlx_cols[i]:
            st.markdown(f"### {role.title()}")
            tlx = surveys.get(f"tlx_{role}", {})
            if "error" in tlx:
                st.warning(tlx["error"])
                continue

            overall = tlx.get("overall_workload")
            if overall is not None:
                st.metric("Overall Workload", f"{overall:.1f}/100")

            dims = {k: v for k, v in tlx.items() if k in ["mental", "physical", "temporal", "performance", "effort", "frustration"] and v is not None}
            if dims:
                fig = go.Figure(go.Bar(
                    x=list(dims.keys()),
                    y=list(dims.values()),
                    marker_color=["#e74c3c", "#3498db", "#f39c12", "#2ecc71", "#9b59b6", "#e67e22"],
                ))
                fig.update_layout(title=f"TLX Dimensions ({role.title()})",
                                  yaxis=dict(range=[0, 100]), height=300)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # PSMM
    st.subheader("🧠 Perceived Shared Mental Models")
    psmm_cols = st.columns(2)
    for i, role in enumerate(["director", "matcher"]):
        with psmm_cols[i]:
            st.markdown(f"### {role.title()}")
            psmm = surveys.get(f"psmm_{role}", {})
            if "error" in psmm:
                st.warning(psmm["error"])
                continue

            mcols = st.columns(3)
            mcols[0].metric("Task SMM", psmm.get("task_smm_mean", "?"))
            mcols[1].metric("Team SMM", psmm.get("team_smm_mean", "?"))
            mcols[2].metric("Overall", psmm.get("overall_smm_mean", "?"))

            items = psmm.get("items", {})
            if items:
                item_df = pd.DataFrame([
                    {"Item": k, "Value": v["value"], "Factor": v["factor"]}
                    for k, v in sorted(items.items(), key=lambda x: int(x[0]))
                ])
                fig = px.bar(item_df, x="Item", y="Value", color="Factor",
                             color_discrete_map={"task": "#3498db", "team": "#e74c3c"},
                             title=f"PSMM Items ({role.title()})")
                fig.update_layout(yaxis=dict(range=[1, 7]), height=300)
                st.plotly_chart(fig, use_container_width=True)


# ─── Cross-Trial Trends ─────────────────────────────────────────────

elif view == "📈 Cross-Trial Trends":
    st.title("📈 Cross-Trial Trends")

    if len(trial_names) < 2:
        st.info("Need at least 2 trials for trend analysis.")
        st.stop()

    # Build cross-trial DataFrame
    rows = []
    for tname in trial_names:
        td = summary.get("trials", {}).get(tname, {})
        row = {"trial": tname}

        surveys = td.get("surveys", {})
        for role in ["director", "matcher"]:
            tlx = surveys.get(f"tlx_{role}", {})
            row[f"tlx_{role}"] = tlx.get("overall_workload")
            psmm = surveys.get(f"psmm_{role}", {})
            row[f"psmm_task_{role}"] = psmm.get("task_smm_mean")
            row[f"psmm_team_{role}"] = psmm.get("team_smm_mean")

        hr = td.get("hr", {})
        row["hr_sync_r"] = hr.get("synchrony", {}).get("pearson_r")
        row["hr_dir_mean"] = hr.get("director", {}).get("stats", {}).get("bpm_mean")
        row["hr_mat_mean"] = hr.get("matcher", {}).get("stats", {}).get("bpm_mean")

        path = td.get("path", {})
        row["path_length"] = path.get("path_length_px")
        row["strokes"] = path.get("num_strokes")

        speech = td.get("speech", {})
        for role in ["director", "matcher"]:
            sp = speech.get(role, {})
            row[f"speech_ratio_{role}"] = sp.get("prosody", {}).get("rate", {}).get("speech_ratio")

        ts = surveys.get("trial_success", {})
        row["target_reached"] = 1 if ts.get("target_reached") else 0
        row["path_confidence"] = ts.get("path_confidence")

        rows.append(row)

    df = pd.DataFrame(rows)
    trial_indices = list(range(len(df)))

    # TLX trends
    st.subheader("📊 Workload (NASA-TLX) Across Trials")
    fig = go.Figure()
    if "tlx_director" in df.columns:
        fig.add_trace(go.Scatter(x=df["trial"], y=df["tlx_director"], name="Director TLX",
                                 mode="lines+markers", line=dict(color="#e74c3c")))
    if "tlx_matcher" in df.columns:
        fig.add_trace(go.Scatter(x=df["trial"], y=df["tlx_matcher"], name="Matcher TLX",
                                 mode="lines+markers", line=dict(color="#3498db")))
    fig.update_layout(yaxis_title="Overall Workload (0-100)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # PSMM trends
    st.subheader("🧠 PSMM Across Trials")
    fig = go.Figure()
    for role, color in [("director", "#e74c3c"), ("matcher", "#3498db")]:
        if f"psmm_task_{role}" in df.columns:
            fig.add_trace(go.Scatter(x=df["trial"], y=df[f"psmm_task_{role}"], name=f"Task SMM ({role[:3]})",
                                     mode="lines+markers", line=dict(color=color, dash="solid")))
        if f"psmm_team_{role}" in df.columns:
            fig.add_trace(go.Scatter(x=df["trial"], y=df[f"psmm_team_{role}"], name=f"Team SMM ({role[:3]})",
                                     mode="lines+markers", line=dict(color=color, dash="dash")))
    fig.update_layout(yaxis_title="PSMM Score (1-7)", yaxis=dict(range=[1, 7]), height=400)
    st.plotly_chart(fig, use_container_width=True)

    # HR synchrony trend
    if df["hr_sync_r"].notna().any():
        st.subheader("❤️ HR Synchrony Across Trials")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["trial"], y=df["hr_sync_r"], name="Pearson r",
                                 mode="lines+markers", line=dict(color="#2ecc71", width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(yaxis_title="Pearson r", yaxis=dict(range=[-1, 1]), height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Path metrics
    if df["path_length"].notna().any():
        st.subheader("🗺️ Path Metrics Across Trials")
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Path Length", "Number of Strokes"])
        fig.add_trace(go.Bar(x=df["trial"], y=df["path_length"], name="Path Length",
                             marker_color="#9b59b6"), row=1, col=1)
        fig.add_trace(go.Bar(x=df["trial"], y=df["strokes"], name="Strokes",
                             marker_color="#f39c12"), row=1, col=2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("🔗 Correlation Matrix")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 2:
        corr_df = df[numeric_cols].corr()
        fig = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="Pairwise Correlations")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
