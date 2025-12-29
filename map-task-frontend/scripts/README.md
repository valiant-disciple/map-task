# Voice Analysis Scripts

Utilities for processing Map Task voice data.

## Setup

1. Install Python 3.8+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install FFmpeg (required for audio conversion):
   - Mac: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`

## Usage

1. **Download ZIP**: Get the session ZIP from the Director page after a trial.
2. **Run Analysis**:
   ```bash
   python analyze_voice.py <path_to_zip> --api-key <SMALLEST_API_KEY>
   ```

## Output

The script creates a folder `<session_id>_analyzed/` containing:
- `audio/director_T*.wav`: Converted audio
- `audio/director_T*_analysis.json`: Smallest.ai transcript + local prosody analysis

## Data Format

**analysis.json**:
- `transcript`: Smallest.ai ASR result (text, words, segments, emotions)
- `prosody`:
  - `pitch`: mean, min, max, std (Hz)
  - `energy`: RMS mean/std
  - `speech_duration_sec`: Time detected as speech
