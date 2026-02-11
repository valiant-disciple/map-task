#!/usr/bin/env python3
"""
Map Task Session Post-Processing Pipeline
==========================================
Processes an exported session ZIP file through all analysis pipelines:
  1. Speech: WebM → WAV conversion, Whisper transcription, prosody features
  2. HR: Baseline correction, interpolation, pairwise synchrony
  3. Path: Matcher path vs reference comparison (Fréchet, DTW)
  4. Surveys: NASA-TLX scoring, PSMM scoring, trial success

Usage:
    python process_session.py <session_zip_or_dir> [options]

Options:
    --output-dir DIR       Output directory (default: data/<session_id>/)
    --whisper-model MODEL  Whisper model: tiny|base|small|medium|large (default: base)
    --skip-speech          Skip speech processing
    --skip-hr              Skip HR processing
    --skip-path            Skip path analysis
    --skip-surveys         Skip survey processing
    --hr-correction METHOD HR baseline correction: subtract|ratio (default: subtract)
    --hr-hz FLOAT          HR resampling rate in Hz (default: 1.0)
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime

# Add parent dir to path for pipeline imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.speech import process_trial_audio
from pipeline.hr import process_trial_hr
from pipeline.path_analysis import process_trial_path
from pipeline.surveys import process_trial_surveys, extract_demographics, extract_debrief


def extract_zip(zip_path: str, output_dir: str) -> str:
    """Extract session ZIP to output_dir. Returns extraction root."""
    print(f"📦 Extracting {zip_path} → {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)
    return output_dir


def find_trial_dirs(session_dir: str) -> list:
    """Find all trial directories (T01, T02, ...) in the session."""
    trials_dir = os.path.join(session_dir, "trials")
    if not os.path.isdir(trials_dir):
        print(f"⚠️  No 'trials/' directory found in {session_dir}")
        return []

    dirs = []
    for name in sorted(os.listdir(trials_dir)):
        trial_path = os.path.join(trials_dir, name)
        if os.path.isdir(trial_path) and name.startswith("T"):
            dirs.append(trial_path)
    return dirs


def load_session_meta(session_dir: str) -> dict:
    """Load session.json metadata."""
    meta_path = os.path.join(session_dir, "session", "session.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def load_all_events(session_dir: str) -> list:
    """Load the global events.json."""
    events_path = os.path.join(session_dir, "session", "events.json")
    if os.path.exists(events_path):
        with open(events_path) as f:
            return json.load(f)
    return []


def process_session(
    input_path: str,
    output_dir: str | None = None,
    whisper_model: str = "base",
    skip_speech: bool = False,
    skip_hr: bool = False,
    skip_path: bool = False,
    skip_surveys: bool = False,
    hr_correction: str = "subtract",
    hr_hz: float = 1.0,
) -> dict:
    """
    Main processing function. Runs all pipelines on a session.
    Returns a summary dict.
    """
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"🔬 Map Task Post-Processing Pipeline")
    print(f"{'='*60}")

    # Determine if input is ZIP or directory
    is_zip = input_path.endswith(".zip") and os.path.isfile(input_path)
    session_id = os.path.splitext(os.path.basename(input_path))[0]
    if session_id.startswith("map_task_session_"):
        session_id = session_id.replace("map_task_session_", "")

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", session_id)

    if is_zip:
        session_dir = extract_zip(input_path, output_dir)
    else:
        session_dir = input_path
        output_dir = input_path

    # Load metadata
    meta = load_session_meta(session_dir)
    events = load_all_events(session_dir)
    trial_dirs = find_trial_dirs(session_dir)

    print(f"\n📋 Session: {meta.get('session', {}).get('id', session_id)}")
    print(f"   Config: mapSet={meta.get('config', {}).get('mapSet', '?')}, "
          f"trials={meta.get('config', {}).get('trialTotal', '?')}, "
          f"warmup={meta.get('config', {}).get('warmupCount', '?')}")
    print(f"   Data trials found: {len(trial_dirs)}")
    print(f"   Total events: {len(events)}")

    # Extract session-level data
    demographics = extract_demographics(events)
    debrief = extract_debrief(events)

    # Process each trial
    summary = {
        "session_id": meta.get("session", {}).get("id", session_id),
        "processed_at": start_time.isoformat(),
        "config": meta.get("config", {}),
        "participants": meta.get("participants", []),
        "demographics": demographics,
        "debrief": debrief,
        "trials": {},
    }

    for trial_dir in trial_dirs:
        trial_name = os.path.basename(trial_dir)
        print(f"\n{'─'*50}")
        print(f"📝 Processing {trial_name}")
        print(f"{'─'*50}")

        trial_result = {"trial_dir": trial_name}

        # 1. Speech
        if not skip_speech:
            print(f"\n  🎤 Speech Analysis:")
            try:
                trial_result["speech"] = process_trial_audio(
                    trial_dir, whisper_model=whisper_model
                )
            except Exception as e:
                print(f"  ❌ Speech pipeline error: {e}")
                trial_result["speech"] = {"error": str(e)}
        else:
            print(f"  ⏭  Speech: skipped")

        # 2. HR
        if not skip_hr:
            print(f"\n  ❤️  HR Analysis:")
            try:
                trial_result["hr"] = process_trial_hr(
                    trial_dir, correction_method=hr_correction, target_hz=hr_hz
                )
            except Exception as e:
                print(f"  ❌ HR pipeline error: {e}")
                trial_result["hr"] = {"error": str(e)}
        else:
            print(f"  ⏭  HR: skipped")

        # 3. Path
        if not skip_path:
            print(f"\n  🗺️  Path Analysis:")
            try:
                # TODO: Load reference path from director map when available
                trial_result["path"] = process_trial_path(trial_dir, reference_path=None)
            except Exception as e:
                print(f"  ❌ Path pipeline error: {e}")
                trial_result["path"] = {"error": str(e)}
        else:
            print(f"  ⏭  Path: skipped")

        # 4. Surveys
        if not skip_surveys:
            print(f"\n  📊 Survey Processing:")
            try:
                trial_result["surveys"] = process_trial_surveys(trial_dir)
            except Exception as e:
                print(f"  ❌ Survey pipeline error: {e}")
                trial_result["surveys"] = {"error": str(e)}
        else:
            print(f"  ⏭  Surveys: skipped")

        summary["trials"][trial_name] = trial_result

    # Save overall summary
    elapsed = (datetime.now() - start_time).total_seconds()
    summary["processing_time_sec"] = round(elapsed, 2)

    summary_path = os.path.join(output_dir, "processing_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"✅ Processing complete in {elapsed:.1f}s")
    print(f"   Summary saved to: {summary_path}")
    print(f"{'='*60}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Map Task Session Post-Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Session ZIP file or extracted directory")
    parser.add_argument("--output-dir", help="Output directory (default: data/<session_id>/)")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--skip-speech", action="store_true", help="Skip speech processing")
    parser.add_argument("--skip-hr", action="store_true", help="Skip HR processing")
    parser.add_argument("--skip-path", action="store_true", help="Skip path analysis")
    parser.add_argument("--skip-surveys", action="store_true", help="Skip survey processing")
    parser.add_argument("--hr-correction", default="subtract",
                        choices=["subtract", "ratio"],
                        help="HR baseline correction method (default: subtract)")
    parser.add_argument("--hr-hz", type=float, default=1.0,
                        help="HR resampling rate in Hz (default: 1.0)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input not found: {args.input}")
        sys.exit(1)

    process_session(
        input_path=args.input,
        output_dir=args.output_dir,
        whisper_model=args.whisper_model,
        skip_speech=args.skip_speech,
        skip_hr=args.skip_hr,
        skip_path=args.skip_path,
        skip_surveys=args.skip_surveys,
        hr_correction=args.hr_correction,
        hr_hz=args.hr_hz,
    )


if __name__ == "__main__":
    main()
