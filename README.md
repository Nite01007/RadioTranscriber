# RadioTranscriber

**My first ever coding project — please be kind!**  
I'm **not** a programmer (at all). This tool was built entirely by describing what I wanted to generative AIs, iterating on their suggestions, and testing/debugging over many sessions. My AI buddy says it's proof that AI-assisted development can take a complete beginner surprisingly far. Feedback, issues, and pull requests are **very welcome**!

A real-time transcription tool for public safety radio feeds (e.g., Broadcastify streams) using OpenAI Whisper large-v3 via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 INT8). Designed for long-running, low-maintenance operation with daily log rotation, robust audio processing, and strong hallucination filtering.

**Important note**: This script is **heavily tuned** to the patterns of my local public safety radio feed (Belchertown, MA area). Unit IDs, dispatch phrasing, alert tones, and filters are customized for that system. It works well on similar feeds, but you will likely need to tweak the prompt, VAD settings, or cleanup rules in `config.yaml` to match your local radio style.

## Features

* Live streaming from authenticated Broadcastify feeds
* High-pass filtering to reduce low-frequency rumble/static
* Percentile-based normalization to handle squelch pops without crushing quiet speech
* WebRTC VAD for reliable speech detection in noisy radio environments
* **faster-whisper** large-v3 transcription with CTranslate2 INT8 quantization (~3-4x faster than openai-whisper on CPU, same model weights)
* Configurable beam search with patience control for accuracy/speed tradeoff
* **MQTT publishing** of transcription state for home automation integration (optional, configurable)
* Powerful hallucination guards:
  + Block full-line credit/caption hallucinations
  + Discard Whisper meta-description artifacts (`Sound effects.`, `Sound of gunfire.`, etc.)
  + Discard dot-only and BANG noise segments
  + Repetition cascade detection: truncate tail cascades, discard full-line cascades, recover content after head cascades
  + Truncate common static-induced endings
  + Replace alert tone or noise hallucinations with `[beeps]` or `[noise]`
  + Spoken-to-unit-ID mapping and hyphen normalization (fully configurable)
* Daily log rollover with clear markers (`[STARTED]`, `[ROLLOVER]`, `[STOPPED]`)
* Memory management with periodic garbage collection for multi-day runs
* Centralized configuration via a single, easy-to-edit `config.yaml`

## Requirements

* Python 3.8+
* ffmpeg (must be in your PATH)
* A Broadcastify premium account (for direct stream access)

## Installation & Setup

1. Clone the repository:
```
   git clone https://github.com/Nite01007/RadioTranscriber.git
   cd RadioTranscriber
```

2. Create and activate a virtual environment (recommended):
```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
   pip install numpy scipy faster-whisper webrtcvad pyyaml paho-mqtt
```

   > **Note:** `faster-whisper` replaces `openai-whisper` and `torch` from earlier versions. If upgrading from a previous install, uninstall those first: `pip uninstall openai-whisper torch`

4. Set up configuration:

   * Copy `config.yaml.example` to `config.yaml` and fill in your values:
     + Broadcastify credentials
     + Feed number and description
     + Whisper model size, prompt, beam/patience settings
     + VAD aggressiveness, min speech length, silence limit
     + MQTT broker settings (optional — set `enabled: false` to disable)
     + Hallucination block phrases, cutoff phrases, unit mappings, etc.

5. (Recommended) Protect your credentials:

   * `config.yaml` is already in `.gitignore`, but verify it is **not tracked** before any push:
```
     git ls-files config.yaml
```
     This should return nothing. If it returns `config.yaml`, run `git rm --cached config.yaml` immediately.

6. Run the transcriber:
```
   python radiotranscriber.py
```

## Output

Transcriptions are saved to daily log files (e.g. `transcription_Belchertown_2025-12-28.log`)

Example log entry:
```
[12:44:52] (21.0s) Central Station 52, stand by for the medical — 123 Main Street...
```

## MQTT Integration

When enabled, the transcriber publishes each transcript line to an MQTT topic for use with home automation platforms (Home Assistant, Node-RED, etc.). Configure broker, port, topic, and credentials in the `mqtt` section of `config.yaml`. Set `enabled: false` to disable entirely with no performance impact.

## Customization (All in config.yaml)

Everything tunable is in one file — no editing the main script needed.

| Section | What You Can Change | Examples / Tips |
|---|---|---|
| `credentials` | Broadcastify username/password | Keep secure! Never commit this file |
| `feed_specific` | Feed number, description, output folder | Output folder auto-created if missing |
| `vad_and_silence` | VAD aggressiveness, min speech seconds, silence limit | Lower VAD = catches more borderline audio; downstream filters handle noise |
| `tuning` | Model size, language, initial prompt, beam size, patience, no-speech threshold, normalization | `beam_size: 10-12` recommended; `patience: 2.0` improves accuracy on ambiguous audio |
| `mqtt` | Broker host/port, topic, credentials, enabled flag | Set `enabled: false` to disable |
| `post_generation_cleanup` | Block phrases, cutoff phrases, unit mapping dict, normalization regex/prefix | Add your local unit phrases here! |

**Tip**: The `initial_prompt` guides Whisper heavily — include your common units, locations, and agencies. Keep it under ~224 tokens. Rebuild it periodically using frequency analysis of your transcript logs to add high-volume addresses and remove dead entries.

## Performance Notes

* Uses **faster-whisper** with INT8 quantization for ~3-4x CPU speedup over openai-whisper at identical accuracy
* Tested on CPU-only hardware (Intel Core i5-3470, no GPU) at ~1.5-2x real-time for typical 15-30s segments
* `beam_size: 12` and `patience: 2.0` are the recommended quality settings for CPU-only deployments with this hardware class
* GPU support available via `compute_type="float16"` — change the model initialization in `radiotranscriber.py`

## Contributing

This project is beginner-friendly and open to improvements!
If you adapt it for your own feed:

* Fork and tweak `config.yaml` for your local dispatch patterns
* If your changes (better defaults, new filters, CLI flags) would help others → open a pull request!
* Issues welcome: bugs, feature ideas, or "it works on my feed now!" success stories

## License

MIT License — feel free to fork, modify, and share.
