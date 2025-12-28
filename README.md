# RadioTranscriber

**My first ever coding project — please be kind!**  
I'm **not** a programmer (at all). This entire tool was built by describing what I wanted to generative AIs, iterating on their code suggestions, and testing/debugging over many sessions. My AI buddy suggests it's a testament to how far AI-assisted development can take a complete beginner. Feedback, issues, and pull requests are very welcome!

A real-time transcription tool for public safety radio feeds (e.g., Broadcastify streams) using OpenAI Whisper (large-v3). Designed for long-running, low-maintenance operation with daily log rotation, robust audio processing, and hallucination filtering.

**Important note**: This script is **heavily tuned** to the patterns of my local public safety radio feed (Belchertown, MA area). Unit IDs, dispatch phrasing, alert tones, and hallucination filters are customized for that system. It should work on similar feeds, but you may need to adjust the prompt, VAD settings, or filters to match your local radio style.

## Features

- **Live streaming** from authenticated Broadcastify feeds
- **Stateful high-pass filtering** to reduce low-frequency rumble/static
- **Robust percentile normalization** to handle loud squelch pops without crushing quiet speech
- **WebRTC VAD** (aggressiveness=3) for reliable speech detection in noisy radio environments
- **Whisper large-v3** transcription with beam search for high accuracy
- **Hallucination guards**:
  - Block full-line credit/caption hallucinations
  - Truncate common static-induced endings (e.g., "Thank you for watching...")
  - Replace alert tone hallucinations with `[beeps]` or `[noise]`
- **Daily log rollover** with clear `[STARTED]` / `[ROLLOVER]` / `[STOPPED]` markers
- **Memory management** with periodic garbage collection for extended runs
- **Configurable** via separate `config.py` (keeps credentials out of main script)

## Requirements

- Python 3.8+
- ffmpeg (must be in PATH)
- A Broadcastify premium account (for direct stream access)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nite01007/RadioTranscriber.git
   cd RadioTranscriber
   
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install dependencies:
   ```bash
   pip install numpy scipy torch openai-whisper webrtcvad

6. Create config.py in the project root (example):
   ```bash
   USERNAME = "your_broadcastify_username"
   PASSWORD = "your_broadcastify_password"
   FEED_NUMBER = "46018"  # e.g., Belchertown feed
   FEED_DESCRIPTION = "Belchertown"  # Used in log filenames

8. Run the transcriber:
   ```bash
   python radiotransriber.py

## Output

Transcriptions are saved to daily log files: (e.g. transcription_Belchertown_2025-12-28.log)

Example log entry:  
```
[12:44:52] (21.0s) Central Station 52, stand by for the medical — 123 Main Street...
```

## Customization

- Change `MIN_SPEECH_SECONDS` in the script to skip more short noise bursts.
- Adjust VAD aggressiveness or other parameters as needed.
- Add more phrases to `FULL_BLOCK_PHRASES` or `CUTOFF_PHRASES` for new hallucinations.
- Modify the `INITIAL_PROMPT` to better match your local dispatch terminology.  
  **Note**: Whisper only uses the last ~224 tokens of the initial prompt. Keep it concise to ensure all instructions are applied (current prompt is trimmed to fit).

## Notes

- This tool is for personal, non-commercial use. Respect Broadcastify's terms of service.
- Runs well on a potato (tested on CPU; GPU optional for faster transcription).
- Memory usage is stable for multi-day runs thanks to periodic cleanup.

## License

MIT License — feel free to fork, modify, and share.
