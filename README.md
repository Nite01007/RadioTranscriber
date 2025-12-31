# RadioTranscriber

**My first ever coding project — please be kind!**  
I'm **not** a programmer (at all). This tool was built entirely by describing what I wanted to generative AIs, iterating on their suggestions, and testing/debugging over many sessions. My AI buddy says it's proof that AI-assisted development can take a complete beginner surprisingly far. Feedback, issues, and pull requests are **very welcome**!

A real-time transcription tool for public safety radio feeds (e.g., Broadcastify streams) using OpenAI Whisper (large-v3). Designed for long-running, low-maintenance operation with daily log rotation, robust audio processing, and strong hallucination filtering.

**Important note**: This script is **heavily tuned** to the patterns of my local public safety radio feed (Belchertown, MA area). Unit IDs, dispatch phrasing, alert tones, and filters are customized for that system. It works well on similar feeds, but you will likely need to tweak the prompt, VAD settings, or cleanup rules in `config.yaml` to match your local radio style.

## Features

- Live streaming from authenticated Broadcastify feeds
- High-pass filtering to reduce low-frequency rumble/static
- Percentile-based normalization to handle squelch pops without crushing quiet speech
- WebRTC VAD for reliable speech detection in noisy radio environments
- Whisper large-v3 transcription with configurable beam search
- Powerful hallucination guards:
  - Block full-line credit/caption hallucinations
  - Truncate common static-induced endings
  - Replace alert tone or noise hallucinations with `[beeps]` or `[noise]`
  - Spoken-to-unit-ID mapping and hyphen normalization (fully configurable!)
- Daily log rollover with clear markers (`[STARTED]`, `[ROLLOVER]`, `[STOPPED]`)
- Memory management with periodic garbage collection for multi-day runs
- Centralized configuration via a single, easy-to-edit `config.yaml`

## Requirements

- Python 3.8+
- ffmpeg (must be in your PATH)
- A Broadcastify premium account (for direct stream access)

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Nite01007/RadioTranscriber.git
   cd RadioTranscriber
   
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
   ```bash
   pip install numpy scipy torch openai-whisper webrtcvad pyyaml

4. Set up configuration
    - Open `config.yaml` in any text editor and fill in your values:
        - Broadcastify credentials
        - Feed number and description
        - Whisper model size, prompt, beam/best-of settings
        - VAD aggressiveness, min speech length, silence limit
        - Hallucination block phrases, cutoff phrases, unit mappings, etc.

5.  (Recommended) Protect your credentials by adding `config.yaml` to `.gitignore`:
    - Create or edit the `.gitignore` file in your project root
    - Add these lines:
      ```text
         config.yaml
         *.log
      ```
    - This prevents accidentally committing your Broadcastify username/password or log files

7. Run the transcriber:
   ```.gitignore
      config.yaml
      *.log

## Output

Transcriptions are saved to daily log files: (e.g. transcription_Belchertown_2025-12-28.log)

Example log entry:  
```
[12:44:52] (21.0s) Central Station 52, stand by for the medical — 123 Main Street...
```

## Customization (All in config.yaml)

Everything tunable is now in one file — no more editing the main script!

| Section                  | What You Can Change                                                                 | Examples / Tips                              |
|--------------------------|-------------------------------------------------------------------------------------|----------------------------------------------|
| `credentials`            | Broadcastify username/password                                                      | Keep secure! Never commit this file          |
| `feed_specific`          | Feed number, description, output folder                                             | Output folder auto-created if missing        |
| `vad_and_silence`        | VAD aggressiveness, min speech seconds, silence limit                               | Higher VAD = fewer false positives on noise  |
| `tuning`                 | Model size, language, initial prompt, beam size, best-of, no-speech threshold, normalization percentile | Larger model = better accuracy but slower    |
| `post_generation_cleanup`| Block phrases, cutoff phrases, unit mapping dict, normalization regex/prefix        | Add your local unit phrases here!            |

**Tip**: The `initial_prompt` guides Whisper heavily — include your common units, locations, and agencies. Keep it under ~224 tokens for best results.

## Notes

- This tool is for personal, non-commercial use. Respect Broadcastify's terms of service.
- Runs well on a potato (tested on CPU; GPU optional for faster transcription).
- Memory usage is stable for multi-day runs thanks to periodic cleanup.

## Contributing

This project is beginner-friendly and open to improvements!
If you adapt it for your own feed:
- Fork and tweak config.yaml for your local dispatch patterns
- If your changes (better defaults, new filters, CLI flags) would help others → open a pull request!
- Issues welcome: bugs, feature ideas, or "it works on my feed now!" success stories

## License

MIT License — feel free to fork, modify, and share.
