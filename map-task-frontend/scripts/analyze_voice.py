import argparse
import os
import json
import zipfile
import shutil
import tempfile
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from smallestai.waves import WavesClient

def analyze_audio(file_path, api_key):
    """
    Process audio file:
    1. Transcribe with Smallest.ai
    2. Extract prosody with librosa
    """
    print(f"Processing {file_path}...")
    
    # --- 1. Smallest.ai Transcription ---
    print("  > Calling Smallest.ai API...")
    try:
        client = WavesClient(api_key=api_key)
        # Assuming analyze/transcribe method based on SDK. 
        # Note: Adjust method name if SDK differs (e.g. client.transcribe())
        transcript_result = client.transcribe(
            file_path,
            language="en",
            diarize=True,
            emotion_detection=True,
            age_detection=True,
            gender_detection=True,
            word_timestamps=True
        )
    except Exception as e:
        print(f"  ! API Error: {e}")
        transcript_result = {"error": str(e)}

    # --- 2. Local Prosody Analysis ---
    print("  > Extracting local prosody features...")
    y, sr = librosa.load(file_path)
    
    # Pitch (F0)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_clean = f0[~np.isnan(f0)]
    
    # Energy (RMS)
    rms = librosa.feature.rms(y=y)[0]
    
    # Speaking Rate (simple estimate based on pauses)
    # Silence detection
    non_silent_intervals = librosa.effects.split(y, top_db=20)
    speech_duration = sum([end - start for start, end in non_silent_intervals]) / sr
    
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    prosody_result = {
        "duration_sec": total_duration,
        "speech_duration_sec": speech_duration,
        "pitch": {
            "mean_hz": float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0,
            "min_hz": float(np.min(f0_clean)) if len(f0_clean) > 0 else 0,
            "max_hz": float(np.max(f0_clean)) if len(f0_clean) > 0 else 0,
            "std_hz": float(np.std(f0_clean)) if len(f0_clean) > 0 else 0
        },
        "energy": {
            "mean_rms": float(np.mean(rms)),
            "std_rms": float(np.std(rms))
        }
    }

    return {
        "transcript": transcript_result,
        "prosody": prosody_result
    }

def process_zip(zip_path, api_key):
    session_id = os.path.splitext(os.path.basename(zip_path))[0]
    output_dir = f"{session_id}_analyzed"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print(f"Extracting to {output_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(output_dir)
        
    # Find audio files
    audio_files = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith('.webm'):
                audio_files.append(os.path.join(root, f))
                
    print(f"Found {len(audio_files)} audio files.")
    
    for webm_path in audio_files:
        # Convert to WAV for compatibility
        wav_path = webm_path.replace('.webm', '.wav')
        print(f"Converting {webm_path} to WAV...")
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio.export(wav_path, format="wav")
        
        # Analyze
        result = analyze_audio(wav_path, api_key)
        
        # Save JSON
        json_path = webm_path.replace('.webm', '_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        print(f"Saved analysis to {json_path}")

    print("Done! Analysis complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Map Task Voice Data")
    parser.add_argument("zip_file", help="Path to session ZIP file")
    parser.add_argument("--api-key", required=True, help="Smallest.ai API Key")
    
    args = parser.parse_args()
    process_zip(args.zip_file, args.api_key)
