"""
Speech Processing Pipeline
- Convert WebM → WAV
- Transcribe with OpenAI Whisper (local, no API needed)
- Extract prosody features with librosa (F0, RMS energy, speaking rate, pauses)
"""

import os
import json
import numpy as np
import librosa
from pydub import AudioSegment


def convert_webm_to_wav(webm_path: str) -> str:
    """Convert WebM audio to WAV. Returns the output path."""
    wav_path = webm_path.rsplit(".", 1)[0] + ".wav"
    if os.path.exists(wav_path):
        return wav_path
    try:
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio.export(wav_path, format="wav")
        print(f"  [speech] Converted {os.path.basename(webm_path)} → WAV")
    except Exception as e:
        print(f"  [speech] Conversion failed for {webm_path}: {e}")
        raise
    return wav_path


def transcribe_whisper(wav_path: str, model_name: str = "base") -> dict:
    """
    Transcribe audio using OpenAI Whisper (runs locally).
    Returns dict with 'text', 'segments' (with timestamps), 'language'.
    """
    try:
        import whisper
    except ImportError:
        print("  [speech] Whisper not installed. pip install openai-whisper")
        return {"text": "", "segments": [], "language": "unknown", "error": "whisper not installed"}

    print(f"  [speech] Transcribing with Whisper ({model_name})...")
    model = whisper.load_model(model_name)
    result = model.transcribe(wav_path, word_timestamps=True)

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "id": seg["id"],
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
            "words": [
                {
                    "word": w["word"].strip(),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                    "probability": round(w.get("probability", 0), 4),
                }
                for w in seg.get("words", [])
            ],
        })

    return {
        "text": result.get("text", "").strip(),
        "language": result.get("language", "unknown"),
        "segments": segments,
    }


def extract_prosody(wav_path: str, frame_length: int = 2048) -> dict:
    """
    Extract prosody features from audio.
    Returns pitch, energy, speaking rate, pauses, and time-series data.
    """
    print(f"  [speech] Extracting prosody features...")
    y, sr = librosa.load(wav_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # 1. Pitch (F0) via pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
        sr=sr, frame_length=frame_length,
    )
    f0_times = librosa.times_like(f0, sr=sr, hop_length=frame_length // 4)
    f0_clean = f0[~np.isnan(f0)]

    pitch_stats = {
        "mean_hz": float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0,
        "std_hz": float(np.std(f0_clean)) if len(f0_clean) > 0 else 0,
        "min_hz": float(np.min(f0_clean)) if len(f0_clean) > 0 else 0,
        "max_hz": float(np.max(f0_clean)) if len(f0_clean) > 0 else 0,
        "range_hz": float(np.ptp(f0_clean)) if len(f0_clean) > 0 else 0,
    }

    # 2. Energy (RMS)
    rms = librosa.feature.rms(y=y, frame_length=frame_length)[0]
    rms_times = librosa.times_like(rms, sr=sr, hop_length=frame_length // 4)
    energy_stats = {
        "mean_rms": float(np.mean(rms)),
        "std_rms": float(np.std(rms)),
        "max_rms": float(np.max(rms)),
    }

    # 3. Speaking rate & pause detection
    non_silent = librosa.effects.split(y, top_db=25)
    speech_frames = sum(e - s for s, e in non_silent)
    speech_sec = speech_frames / sr
    silence_sec = duration - speech_sec

    # Detect individual pauses (gaps between voiced segments)
    pauses = []
    for i in range(1, len(non_silent)):
        gap_start = non_silent[i - 1][1] / sr
        gap_end = non_silent[i][0] / sr
        gap_dur = gap_end - gap_start
        if gap_dur > 0.3:  # Only count pauses > 300ms
            pauses.append({
                "start": round(gap_start, 3),
                "end": round(gap_end, 3),
                "duration": round(gap_dur, 3),
            })

    rate_stats = {
        "total_duration_sec": round(duration, 3),
        "speech_duration_sec": round(speech_sec, 3),
        "silence_duration_sec": round(silence_sec, 3),
        "speech_ratio": round(speech_sec / duration, 4) if duration > 0 else 0,
        "num_pauses": len(pauses),
        "mean_pause_sec": round(np.mean([p["duration"] for p in pauses]), 3) if pauses else 0,
    }

    # Time series (downsampled for storage)
    step = max(1, len(f0_times) // 500)
    pitch_ts = [
        {"t": round(float(f0_times[i]), 3), "hz": round(float(f0[i]), 2) if not np.isnan(f0[i]) else None}
        for i in range(0, len(f0_times), step)
    ]
    energy_ts = [
        {"t": round(float(rms_times[i]), 3), "rms": round(float(rms[i]), 6)}
        for i in range(0, len(rms_times), step)
    ]

    return {
        "pitch": {"stats": pitch_stats, "timeseries": pitch_ts},
        "energy": {"stats": energy_stats, "timeseries": energy_ts},
        "rate": rate_stats,
        "pauses": pauses,
    }


def process_audio_file(
    webm_path: str,
    whisper_model: str = "base",
    skip_transcription: bool = False,
) -> dict:
    """Full speech pipeline for a single audio file."""
    wav_path = convert_webm_to_wav(webm_path)

    result = {"wav_path": wav_path}

    # Transcription
    if not skip_transcription:
        result["transcript"] = transcribe_whisper(wav_path, model_name=whisper_model)
    else:
        result["transcript"] = {"text": "", "segments": [], "language": "skipped"}

    # Prosody
    result["prosody"] = extract_prosody(wav_path)

    return result


def process_trial_audio(trial_dir: str, whisper_model: str = "base", skip_transcription: bool = False) -> dict:
    """
    Process all audio files in a trial directory.
    Expects: trial_dir/audio/director_T*.webm, matcher_T*.webm
    """
    audio_dir = os.path.join(trial_dir, "audio")
    if not os.path.isdir(audio_dir):
        return {}

    results = {}
    for fname in sorted(os.listdir(audio_dir)):
        if not fname.endswith(".webm"):
            continue
        role = "director" if "director" in fname.lower() else "matcher"
        webm_path = os.path.join(audio_dir, fname)
        print(f"  [speech] Processing {fname}...")
        try:
            analysis = process_audio_file(webm_path, whisper_model, skip_transcription)
            results[role] = analysis
            # Save per-file JSON
            json_path = webm_path.rsplit(".", 1)[0] + "_analysis.json"
            with open(json_path, "w") as f:
                json.dump(analysis, f, indent=2, default=str)
        except Exception as e:
            print(f"  [speech] ERROR processing {fname}: {e}")
            results[role] = {"error": str(e)}

    return results
