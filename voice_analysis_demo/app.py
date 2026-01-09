import streamlit as st
import os
import json
import tempfile
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pydub import AudioSegment
from smallestai.waves import WavesClient

# Page Config
st.set_page_config(page_title="Voice Analysis Demo", page_icon="🎙️", layout="wide")

st.title("🎙️ Voice Analysis Checker")
st.markdown("Check transcription, emotion, and prosody using **Smallest.ai** + **Librosa**.")

# Sidebar - Settings
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Smallest.ai API Key", value="sk_5a63af8737a305e455ce85bd97358560", type="password")
    
    st.divider()
    st.subheader("Analysis Options")
    enable_diarization = st.checkbox("Speaker Diarization", value=True)
    enable_emotion = st.checkbox("Emotion Detection", value=True)
    enable_prosody = st.checkbox("Local Prosody (Pitch/Energy)", value=True)

# Helper: Save Uploaded/Recorded File
def save_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1] if hasattr(audio_file, 'name') else ".wav") as tmp:
        tmp.write(audio_file.getvalue())
        return tmp.name

# Helper: Convert to WAV if needed (WebM, etc.)
def convert_to_wav(file_path):
    if file_path.endswith('.wav'):
        return file_path
    
    wav_path = file_path.rsplit('.', 1)[0] + ".wav"
    try:
        audio = AudioSegment.from_file(file_path)
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        st.error(f"Error converting audio: {e}")
        return None

# --- ANALYSIS FUNCTIONS ---

def analyze_prosody(wav_path):
    y, sr = librosa.load(wav_path)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 1. Pitch (F0)
    # Using pyin for robust estimation
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        frame_length=2048
    )
    times = librosa.times_like(f0, sr=sr)
    
    # Clean F0 for stats
    f0_clean = f0[~np.isnan(f0)]
    pitch_stats = {
        "mean": float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0,
        "std": float(np.std(f0_clean)) if len(f0_clean) > 0 else 0,
        "min": float(np.min(f0_clean)) if len(f0_clean) > 0 else 0,
        "max": float(np.max(f0_clean)) if len(f0_clean) > 0 else 0
    }

    # 2. Energy (RMS)
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.times_like(rms, sr=sr)
    energy_stats = {
        "mean": float(np.mean(rms)),
        "std": float(np.std(rms))
    }

    # 3. Speaking Rate
    # Simple estimation based on non-silent chunks
    non_silent = librosa.effects.split(y, top_db=20)
    speech_frames = sum([e - s for s, e in non_silent])
    speech_sec = speech_frames / sr
    
    return {
        "duration": duration,
        "speech_sec": speech_sec,
        "pitch": {"stats": pitch_stats, "data": f0, "times": times},
        "energy": {"stats": energy_stats, "data": rms, "times": rms_times}
    }

def analyze_smallest_ai(file_path, api_key, diarize, emotion):
    client = WavesClient(api_key=api_key)
    try:
        # Based on SDK usage
        return client.transcribe(
            file_path,
            language="en",
            # diarize=diarize, # Not supported in SDK 4.1.0 transcribe method
            emotion_detection=emotion,
            age_detection=True,
            gender_detection=True,
            word_timestamps=True
        )
    except Exception as e:
        st.error(f"Smallest.ai API Error: {e}")
        return None

# --- MAIN UI ---

tab1, tab2 = st.tabs(["Upload File", "Record Audio"])

audio_file = None
with tab1:
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'webm', 'm4a'])
    if uploaded_file:
        audio_file = uploaded_file

with tab2:
    recorded_audio = st.audio_input("Record Microphone")
    if recorded_audio:
        audio_file = recorded_audio

if audio_file:
    # Save and prepare
    temp_path = save_audio(audio_file)
    wav_path = convert_to_wav(temp_path)
    
    st.audio(wav_path, format='audio/wav')
    
    if st.button("Run Analysis", type="primary"):
        with st.spinner("Analyzing audio..."):
            
            # 1. Prosody Analysis (Local)
            prosody_data = None
            if enable_prosody:
                prosody_data = analyze_prosody(wav_path)
            
            # 2. Transcription (API)
            transcription_data = analyze_smallest_ai(wav_path, api_key, enable_diarization, enable_emotion)
            
            st.success("Analysis Complete!")
            
            # --- RESULTS DISPLAY ---
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📝 Transcript")
                if transcription_data:
                    # Depending on exact structure of response, standardizing display
                    # Assuming typical format: {"text": "...", "segments": [...]}
                    st.write(transcription_data.get('text', 'No text found.'))
                    
                    if 'segments' in transcription_data:
                        st.divider()
                        for seg in transcription_data['segments']:
                            spk = seg.get('speaker', 'Unknown')
                            text = seg.get('text', '')
                            emo = seg.get('emotion', {})
                            st.markdown(f"**{spk}:** {text}")
                            if emo:
                                # Show top emotion
                                if isinstance(emo, dict):
                                    top_emo = max(emo, key=emo.get)
                                    st.caption(f"🎭 {top_emo} ({emo[top_emo]:.2f})")
                                else:
                                    st.caption(f"🎭 {emo}")

            with col2:
                if prosody_data:
                    st.subheader("📊 Voice Prosody")
                    
                    # Pitch Plot
                    pitch = prosody_data['pitch']
                    fig_pitch = px.line(x=pitch['times'], y=pitch['data'], title="Pitch Contour (F0)", labels={'x': 'Time (s)', 'y': 'Frequency (Hz)'})
                    fig_pitch.update_layout(height=300)
                    st.plotly_chart(fig_pitch, use_container_width=True)
                    
                    # Stats Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Avg Pitch", f"{pitch['stats']['mean']:.1f} Hz")
                    m2.metric("Pitch Range", f"{pitch['stats']['max'] - pitch['stats']['min']:.1f} Hz")
                    m3.metric("Speaking Duration", f"{prosody_data['speech_sec']:.1f}s")
                    
                    # Energy Plot
                    energy = prosody_data['energy']
                    fig_energy = px.line(x=energy['times'], y=energy['data'], title="Energy (Loudness)", labels={'x': 'Time (s)', 'y': 'RMS Energy'})
                    fig_energy.update_layout(height=300)
                    st.plotly_chart(fig_energy, use_container_width=True)

            # Raw Data Tabs
            with st.expander("View Raw JSON Data"):
                st.json({
                    "transcript": transcription_data,
                    "prosody_stats": {
                        "pitch": prosody_data['pitch']['stats'] if prosody_data else None,
                        "energy": prosody_data['energy']['stats'] if prosody_data else None,
                        "speech_sec": prosody_data['speech_sec'] if prosody_data else None
                    }
                })
    
    # Cleanup
    # os.remove(temp_path) 
    # os.remove(wav_path) # Keeping for debugging if needed, usually tempfile handles/OS cleans
