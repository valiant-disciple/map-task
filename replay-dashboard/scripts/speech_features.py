#!/usr/bin/env python3
"""
Speech feature extraction: Whisper ASR + Parselmouth prosody + OpenSMILE eGeMAPSv02.

Usage:
    from speech_features import extract_speech_features, extract_pair_features

Requirements:
    pip install openai parselmouth opensmile numpy
"""

import io
import os
import re
import tempfile
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# ── OpenAI Whisper ASR ──

def _whisper_transcribe(audio_path: str, api_key: str = None) -> Dict[str, Any]:
    """Transcribe audio with Whisper API, returning word-level timestamps."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language="en",
        )
    result = {
        "text": resp.text or "",
        "language": getattr(resp, "language", "en"),
        "duration": getattr(resp, "duration", 0.0),
        "segments": [],
        "words": [],
    }
    for seg in getattr(resp, "segments", []) or []:
        result["segments"].append({
            "start": seg.start if hasattr(seg, "start") else seg.get("start", 0),
            "end": seg.end if hasattr(seg, "end") else seg.get("end", 0),
            "text": seg.text if hasattr(seg, "text") else seg.get("text", ""),
        })
    for w in getattr(resp, "words", []) or []:
        result["words"].append({
            "word": w.word if hasattr(w, "word") else w.get("word", ""),
            "start": w.start if hasattr(w, "start") else w.get("start", 0),
            "end": w.end if hasattr(w, "end") else w.get("end", 0),
        })
    return result


# ── Parselmouth prosody ──

def _parselmouth_features(audio_path: str) -> Dict[str, float]:
    """Extract prosody features using Parselmouth (Praat bindings)."""
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return {}

    try:
        snd = parselmouth.Sound(audio_path)
    except Exception:
        return {}

    feats = {}
    duration = snd.get_total_duration()
    feats["duration_sec"] = duration

    # F0 (pitch)
    pitch = call(snd, "To Pitch", 0.0, 75, 500)
    f0_values = [call(pitch, "Get value in frame", i + 1, "Hertz")
                 for i in range(call(pitch, "Get number of frames"))]
    f0_valid = [v for v in f0_values if v == v and v > 0]  # filter NaN
    if f0_valid:
        feats["f0_mean"] = float(np.mean(f0_valid))
        feats["f0_std"] = float(np.std(f0_valid))
        feats["f0_min"] = float(np.min(f0_valid))
        feats["f0_max"] = float(np.max(f0_valid))
        feats["f0_range"] = feats["f0_max"] - feats["f0_min"]
        feats["f0_median"] = float(np.median(f0_valid))
        feats["f0_coverage"] = len(f0_valid) / max(len(f0_values), 1)
    else:
        for k in ["f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_median", "f0_coverage"]:
            feats[k] = 0.0

    # Intensity
    intensity = call(snd, "To Intensity", 75, 0.0, "yes")
    feats["intensity_mean"] = call(intensity, "Get mean", 0, 0, "dB")
    feats["intensity_std"] = call(intensity, "Get standard deviation", 0, 0)
    feats["intensity_min"] = call(intensity, "Get minimum", 0, 0, "parabolic")
    feats["intensity_max"] = call(intensity, "Get maximum", 0, 0, "parabolic")

    # Jitter & Shimmer (voice quality)
    try:
        point_proc = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        feats["jitter_local"] = call(point_proc, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        feats["jitter_rap"] = call(point_proc, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        feats["jitter_ppq5"] = call(point_proc, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        feats["shimmer_local"] = call([snd, point_proc], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        feats["shimmer_apq3"] = call([snd, point_proc], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        feats["shimmer_apq5"] = call([snd, point_proc], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        for k in ["jitter_local", "jitter_rap", "jitter_ppq5", "shimmer_local", "shimmer_apq3", "shimmer_apq5"]:
            feats[k] = float("nan")

    # HNR (Harmonics-to-Noise Ratio)
    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        feats["hnr_mean"] = call(harmonicity, "Get mean", 0, 0)
        feats["hnr_std"] = call(harmonicity, "Get standard deviation", 0, 0)
    except Exception:
        feats["hnr_mean"] = float("nan")
        feats["hnr_std"] = float("nan")

    # Formants (F1-F4 means)
    try:
        formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        for fi in range(1, 5):
            vals = []
            n_frames = call(formant, "Get number of frames")
            for frame in range(1, n_frames + 1):
                v = call(formant, "Get value at time", fi, call(formant, "Get time from frame number", frame), "Hertz", "Linear")
                if v == v and v > 0:
                    vals.append(v)
            feats[f"f{fi}_mean"] = float(np.mean(vals)) if vals else 0.0
            feats[f"f{fi}_std"] = float(np.std(vals)) if vals else 0.0
    except Exception:
        for fi in range(1, 5):
            feats[f"f{fi}_mean"] = 0.0
            feats[f"f{fi}_std"] = 0.0

    # Replace any NaN with 0.0
    for k, v in feats.items():
        if isinstance(v, float) and (v != v or not np.isfinite(v)):
            feats[k] = 0.0

    return feats


# ── OpenSMILE eGeMAPSv02 ──

def _opensmile_features(audio_path: str) -> Dict[str, float]:
    """Extract eGeMAPSv02 feature set using openSMILE."""
    try:
        import opensmile
    except ImportError:
        return {}

    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        df = smile.process_file(audio_path)
        if df.empty:
            return {}
        row = df.iloc[0]
        return {f"esmile_{col}": float(row[col]) for col in df.columns}
    except Exception:
        return {}


# ── Traditional linguistic features from transcript ──

FILLER_WORDS = {"um", "uh", "uhm", "hmm", "hm", "like", "you know", "i mean", "sort of", "kind of"}
SPATIAL_WORDS = {
    "left", "right", "up", "down", "above", "below", "north", "south", "east", "west",
    "top", "bottom", "straight", "across", "around", "through", "past", "near", "far",
    "towards", "toward", "along", "between", "beside", "next to", "opposite", "corner",
    "edge", "center", "middle", "over", "under", "diagonal", "horizontal", "vertical",
    "clockwise", "counterclockwise", "curve", "turn", "go", "start", "stop", "end",
    "begin", "continue", "follow", "reach", "cross", "loop", "parallel",
}


def _linguistic_features(transcript: Dict[str, Any]) -> Dict[str, float]:
    """Compute traditional linguistic features from Whisper transcript."""
    text = transcript.get("text", "")
    words = transcript.get("words", [])
    duration = transcript.get("duration", 0.0)

    if not text.strip():
        return {
            "word_count": 0, "unique_words": 0, "ttr": 0.0,
            "speech_rate_wpm": 0.0, "filler_count": 0, "filler_rate": 0.0,
            "spatial_word_count": 0, "spatial_density": 0.0,
            "mean_word_duration": 0.0, "total_speech_sec": 0.0,
            "total_pause_sec": 0.0, "pause_count": 0, "mean_pause_sec": 0.0,
            "max_pause_sec": 0.0, "speech_ratio": 0.0,
            "sentence_count": 0, "mean_sentence_length": 0.0,
        }

    tokens = text.lower().split()
    word_count = len(tokens)
    unique = set(tokens)
    ttr = len(unique) / word_count if word_count else 0.0
    speech_rate = (word_count / duration) * 60 if duration > 0 else 0.0

    # Fillers
    text_lower = text.lower()
    filler_count = sum(len(re.findall(r'\b' + re.escape(f) + r'\b', text_lower)) for f in FILLER_WORDS)
    filler_rate = filler_count / word_count if word_count else 0.0

    # Spatial density
    bigrams = [tokens[i] + " " + tokens[i+1] for i in range(len(tokens)-1)]
    spatial_count = sum(1 for t in tokens if t in SPATIAL_WORDS) + sum(1 for b in bigrams if b in SPATIAL_WORDS)
    spatial_density = spatial_count / word_count if word_count else 0.0

    # Pause analysis from word timestamps
    pauses = []
    total_speech = 0.0
    word_durations = []
    for i, w in enumerate(words):
        wd = w.get("end", 0) - w.get("start", 0)
        if wd > 0:
            word_durations.append(wd)
            total_speech += wd
        if i > 0:
            gap = w.get("start", 0) - words[i - 1].get("end", 0)
            if gap > 0.15:  # pause threshold 150ms
                pauses.append(gap)

    total_pause = sum(pauses)
    speech_ratio = total_speech / duration if duration > 0 else 0.0

    # Sentence-level
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = len(sentences)
    mean_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0.0

    return {
        "word_count": word_count,
        "unique_words": len(unique),
        "ttr": ttr,
        "speech_rate_wpm": speech_rate,
        "filler_count": filler_count,
        "filler_rate": filler_rate,
        "spatial_word_count": spatial_count,
        "spatial_density": spatial_density,
        "mean_word_duration": float(np.mean(word_durations)) if word_durations else 0.0,
        "total_speech_sec": total_speech,
        "total_pause_sec": total_pause,
        "pause_count": len(pauses),
        "mean_pause_sec": float(np.mean(pauses)) if pauses else 0.0,
        "max_pause_sec": float(max(pauses)) if pauses else 0.0,
        "speech_ratio": speech_ratio,
        "sentence_count": sentence_count,
        "mean_sentence_length": float(mean_sent_len),
    }


# ── Public API ──

def extract_speech_features(audio_path: str, role: str, trial: int,
                            api_key: str = None) -> Dict[str, Any]:
    """
    Full speech feature extraction for a single audio file.

    Returns dict with keys:
        role, trial, transcript (raw text), words (list),
        + all prosody/linguistic/opensmile features prefixed by source.
    """
    result = {"role": role, "trial": trial, "audio_path": audio_path}

    # 1. Whisper ASR
    try:
        transcript = _whisper_transcribe(audio_path, api_key)
        result["transcript"] = transcript.get("text", "")
        result["asr_duration"] = transcript.get("duration", 0.0)
        result["asr_word_count"] = len(transcript.get("words", []))
        result["words"] = transcript.get("words", [])
        result["segments"] = transcript.get("segments", [])
    except Exception as e:
        result["transcript"] = ""
        result["asr_error"] = str(e)[:200]
        transcript = {"text": "", "words": [], "duration": 0.0}

    # 2. Linguistic features
    ling = _linguistic_features(transcript)
    result.update({f"ling_{k}": v for k, v in ling.items()})

    # 3. Parselmouth prosody
    prosody = _parselmouth_features(audio_path)
    result.update({f"praat_{k}": v for k, v in prosody.items()})

    # 4. OpenSMILE eGeMAPSv02
    smile = _opensmile_features(audio_path)
    result.update(smile)  # already prefixed esmile_

    return result


def extract_pair_features(director_audio: str, matcher_audio: str, trial: int,
                          api_key: str = None) -> Dict[str, Any]:
    """
    Extract dyadic speech features: convergence, turn-taking, overlap.

    Requires both audio files to have been processed by extract_speech_features first
    (or pass raw paths and it will run Whisper).
    """
    d_feats = extract_speech_features(director_audio, "director", trial, api_key)
    m_feats = extract_speech_features(matcher_audio, "matcher", trial, api_key)

    pair = {"trial": trial}

    # Flatten individual features with role prefix
    for k, v in d_feats.items():
        if k not in ("role", "trial", "audio_path", "words", "segments"):
            pair[f"director_{k}"] = v
    for k, v in m_feats.items():
        if k not in ("role", "trial", "audio_path", "words", "segments"):
            pair[f"matcher_{k}"] = v

    # Turn-taking analysis from word timestamps
    d_words = d_feats.get("words", [])
    m_words = m_feats.get("words", [])

    if d_words and m_words:
        # Build speech intervals
        d_intervals = [(w["start"], w["end"]) for w in d_words if w.get("start") is not None]
        m_intervals = [(w["start"], w["end"]) for w in m_words if w.get("start") is not None]

        # Merge word-level intervals into continuous speech segments per speaker
        def _merge_intervals(intervals):
            if not intervals:
                return []
            sorted_iv = sorted(intervals, key=lambda x: x[0])
            merged = [sorted_iv[0]]
            for s, e in sorted_iv[1:]:
                if s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            return merged

        d_merged = _merge_intervals(d_intervals)
        m_merged = _merge_intervals(m_intervals)

        # Count overlaps between merged segments
        overlaps = 0
        overlap_duration = 0.0
        for ds, de in d_merged:
            for ms, me in m_merged:
                ov_start = max(ds, ms)
                ov_end = min(de, me)
                if ov_start < ov_end:
                    overlaps += 1
                    overlap_duration += ov_end - ov_start

        pair["overlap_count"] = overlaps
        pair["overlap_duration_sec"] = overlap_duration

        # Turn count (who speaks after whom)
        all_words = [(w["start"], "d") for w in d_words if w.get("start")] + \
                    [(w["start"], "m") for w in m_words if w.get("start")]
        all_words.sort()
        turns = 0
        prev_speaker = None
        for _, speaker in all_words:
            if speaker != prev_speaker and prev_speaker is not None:
                turns += 1
            prev_speaker = speaker
        pair["turn_count"] = turns

        # Speech dominance ratio (director / total)
        d_total = sum(de - ds for ds, de in d_intervals)
        m_total = sum(me - ms for ms, me in m_intervals)
        total = d_total + m_total
        pair["director_speech_ratio"] = d_total / total if total > 0 else 0.5

        # F0 convergence (difference in mean pitch)
        d_f0 = d_feats.get("praat_f0_mean", 0)
        m_f0 = m_feats.get("praat_f0_mean", 0)
        if d_f0 > 0 and m_f0 > 0:
            pair["f0_convergence"] = abs(d_f0 - m_f0)  # lower = more converged
            pair["f0_ratio"] = min(d_f0, m_f0) / max(d_f0, m_f0)

        # Speech rate convergence
        d_rate = d_feats.get("ling_speech_rate_wpm", 0)
        m_rate = m_feats.get("ling_speech_rate_wpm", 0)
        if d_rate > 0 and m_rate > 0:
            pair["speech_rate_convergence"] = abs(d_rate - m_rate)
            pair["speech_rate_ratio"] = min(d_rate, m_rate) / max(d_rate, m_rate)

        # Vocabulary overlap (Jaccard)
        d_vocab = set(d_feats.get("transcript", "").lower().split())
        m_vocab = set(m_feats.get("transcript", "").lower().split())
        if d_vocab or m_vocab:
            pair["vocab_jaccard"] = len(d_vocab & m_vocab) / len(d_vocab | m_vocab) if (d_vocab | m_vocab) else 0.0

    return pair


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Extract speech features from audio file")
    ap.add_argument("--audio", required=True, help="Path to audio file (WAV/WebM)")
    ap.add_argument("--role", default="unknown", help="Speaker role (director/matcher)")
    ap.add_argument("--trial", type=int, default=0, help="Trial number")
    ap.add_argument("--api-key", default=None, help="OpenAI API key")
    args = ap.parse_args()

    feats = extract_speech_features(args.audio, args.role, args.trial, args.api_key)
    # Remove non-serializable
    for k in ["words", "segments"]:
        if k in feats:
            feats[k] = len(feats[k])
    print(json.dumps(feats, indent=2, default=str))
