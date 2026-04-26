# https://github.com/Nite01007/RadioTranscriber
import subprocess
import numpy as np
from faster_whisper import WhisperModel
import datetime
import time
import sys
import queue
import threading
import select
import scipy.signal as signal
import os
import gc
import re
from collections import Counter
import webrtcvad
import yaml
# Optional MQTT publishing (for Home Assistant dashboard)
try:
    from mqtt_publisher import MqttPublisher
except Exception:
    MqttPublisher = None


# Load configuration
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
# Extract config values
USERNAME = config["credentials"]["username"]
PASSWORD = config["credentials"]["password"]
FEED_NUMBER = config["feed_specific"]["feed_number"]
FEED_DESCRIPTION = config["feed_specific"]["description"]
OUTPUT_FOLDER = config["feed_specific"]["output_folder"]

VAD_AGGRESSIVENESS = config["vad_and_silence"]["vad_aggressiveness"]
MIN_SPEECH_SECONDS = config["vad_and_silence"]["min_speech_seconds"]
SILENCE_LIMIT = config["vad_and_silence"]["silence_limit"]

MODEL_SIZE = config["tuning"]["model_size"]
LANGUAGE = config["tuning"]["language"]
INITIAL_PROMPT = config["tuning"]["initial_prompt"]
BEAM_SIZE = config["tuning"]["beam_size"]
NO_SPEECH_THRESHOLD = config["tuning"]["no_speech_threshold"]
NORMALIZATION_PERCENTILE = config["tuning"]["normalization"]

FULL_BLOCK_PHRASES = config["post_generation_cleanup"]["full_block_phrases"]
CUTOFF_PHRASES = config["post_generation_cleanup"]["cutoff_phrases"]
UNIT_MAPPING = config["post_generation_cleanup"]["unit_mapping"]
UNIT_PATTERN = config["post_generation_cleanup"]["unit_normalization"]["pattern"]
UNIT_PREFIX = config["post_generation_cleanup"]["unit_normalization"]["prefix"]

# --- MQTT (optional) ---
mqtt_pub = None
MQTT_CFG = config.get("mqtt", {})
if MqttPublisher is not None and isinstance(MQTT_CFG, dict) and MQTT_CFG.get("enabled"):
    try:
        mqtt_pub = MqttPublisher(MQTT_CFG)
        print(f"MQTT enabled → publishing to {MQTT_CFG.get('topic_base', '<missing>')}/state")
    except Exception as e:
        mqtt_pub = None
        print(f"MQTT init failed (continuing without MQTT): {e}")

# --- SETTINGS ---
STREAM_URL = f"http://{USERNAME}:{PASSWORD}@audio.broadcastify.com/{FEED_NUMBER}.mp3"
SAMPLE_RATE = 16000
CHUNK_BYTES = 8192
IDLE_THRESHOLD_SECONDS = 600
GC_INTERVAL = 100

BASE_LOG_FILENAME = f"transcription_{FEED_DESCRIPTION}"
LOG_FILE = os.path.join(OUTPUT_FOLDER, f"{BASE_LOG_FILENAME}_{datetime.date.today()}.log")
CURRENT_LOG_DATE = datetime.date.today()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
if not os.path.exists(LOG_FILE):
    open(LOG_FILE, 'a').close()

sos = signal.butter(5, 100 / (SAMPLE_RATE / 2), btype='high', output='sos')
filter_state = np.zeros((sos.shape[0], 2))

vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
VAD_FRAME_MS = 30
VAD_FRAME_BYTES = int(SAMPLE_RATE * (VAD_FRAME_MS / 1000) * 2)

transcription_queue = queue.Queue()
_log_lock = threading.Lock()

# --- INITIALIZATION ---
device = "cpu"
print(f"Loading Whisper model '{MODEL_SIZE}' (faster-whisper, INT8)...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4)
print(f"Model loaded on {device}")

redacted_url = f"http://{USERNAME}:********@audio.broadcastify.com/{FEED_NUMBER}.mp3"
print(f"Streaming {FEED_DESCRIPTION} feed ({FEED_NUMBER}) from: {redacted_url}")
print("   Press 'Q' to quit cleanly")

last_activity_time = time.time()


def get_ffmpeg_stream(url):
    command = [
        'ffmpeg',
        '-reconnect', '1', '-reconnect_streamed', '1', '-reconnect_delay_max', '5',
        '-i', url,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', str(SAMPLE_RATE),
        '-ac', '1',
        '-loglevel', 'quiet',
        '-'
    ]
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

def transcriber_worker(model, device):
    print(f"   [Worker] Transcriber thread started on {device}")
    
    while True:
        try:
            timestamp, audio_data = transcription_queue.get()
            if audio_data is None: 
                break

            duration = len(audio_data) / SAMPLE_RATE
            print(f"Transcribing {duration:.1f}s segment...")
            transcribe_start = time.time()
            
            segments, info = model.transcribe(
                audio_data,
                language=LANGUAGE,
                initial_prompt=INITIAL_PROMPT,
                condition_on_previous_text=False,
                temperature=0.0,
                beam_size=BEAM_SIZE,
                patience=1.5,
                suppress_blank=True,
                no_speech_threshold=NO_SPEECH_THRESHOLD
            )
            segments = list(segments)  # Force evaluation — generator exhausts on first use

            text = " ".join(s.text for s in segments).strip()
            transcribe_time = time.time() - transcribe_start
            print(f"   Done in {transcribe_time:.1f}s")

            original_text = text

            # No-speech prob filter
            no_speech_prob = max((s.no_speech_prob for s in segments), default=0)
            if no_speech_prob > NO_SPEECH_THRESHOLD:
                print(f"   (Discarded non-speech segment: prob {no_speech_prob:.2f})")
                transcription_queue.task_done()
                continue

            # Dot-only discard filter (231 confirmed occurrences)
            if re.fullmatch(r'[.\s]+', text):
                print(f"   (Discarded dot-only segment: {repr(text)})")
                transcription_queue.task_done()
                continue

            # BANG discard filter (106 confirmed occurrences — mic pops / MDT tones)
            stripped_bang = re.sub(r'[\W_]+', '', text).upper()
            if stripped_bang and re.fullmatch(r'(BANG)+', stripped_bang):
                print(f"   (Discarded BANG segment: {repr(text)})")
                transcription_queue.task_done()
                continue

            # Beep hallucination replacement
            if duration < 10.0:
                beep_patterns = [
                    "BEEE", "BEEEE", "EEEE", "BEEP", "BEEEP", 
                    "BEEEEEEEE", "EEEEEEEE", 
                    "AAAA", "AAAAA", "AAAAAA", "A A A", 
                    "AAAAAAAAA", "AAAAAAAAAA"
                ]
                upper_text = text.upper()
                if any(pattern in upper_text for pattern in beep_patterns) and len(text) > 10:
                    text = "[beeps]"
                    print(f"   (Replaced alert tone: {original_text} → [beeps])")

            # Long single-char run → [noise]
            if re.search(r'([A-Z])\1{10,}', text.upper()):
                text = "[noise]"
                print(f"   (Replaced noise run: {original_text} → [noise])")

            # De-duplicate repeated unit calls
            words = text.split()
            if len(words) > 3:
                common = Counter(words).most_common(1)
                if common and len(common[0][0]) <= 10 and common[0][1] > len(words) // 2:
                    text = common[0][0]
                    print(f"   (De-duplicated repetition: {original_text} → {text})")

            # Repetition cascade filter
            # Part A: physically impossible speech rate
            if duration > 0 and len(text.split()) / duration > 8.0:
                print(f"   (Discarded impossible speech rate: {len(text.split())/duration:.1f} w/s: {text})")
                transcription_queue.task_done()
                continue

            # Part B: detect word/phrase repeating 4+ consecutive times
            words = text.split()
            cascade_match = None
            for phrase_len in range(1, 4):
                for i in range(len(words) - phrase_len * 3):
                    phrase = words[i:i + phrase_len]
                    count = 1
                    j = i + phrase_len
                    while j + phrase_len <= len(words) and words[j:j + phrase_len] == phrase:
                        count += 1
                        j += phrase_len
                    if count >= 4:
                        cascade_match = (i, phrase_len, phrase)
                        break
                if cascade_match:
                    break
            if cascade_match:
                start_idx, phrase_len, phrase = cascade_match
                if start_idx >= 4:
                    text = ' '.join(words[:start_idx + phrase_len])
                    lower_text = text.lower()
                    print(f"   (Truncated cascade: {original_text} → {text})")
                else:
                    print(f"   (Discarded cascade: {original_text})")
                    transcription_queue.task_done()
                    continue

            # Map spoken numbers to letter units
            lower_text = text.lower()
            for spoken, letter in UNIT_MAPPING.items():
                if spoken in lower_text:
                    text = re.sub(r'\b' + re.escape(spoken) + r'\b', letter, text, flags=re.IGNORECASE)
                    print(f"   (Mapped unit: {original_text} → {text})")

            # Normalize hyphens in unit IDs (e.g., 5-2-A-1 → 52A1)
            # Targets 52-prefixed units with letters (A,E,L,B,BS,BL) + digits
            def normalize_unit(match):
                letter = match.group(1).upper()
                num = match.group(2)
                return f"{UNIT_PREFIX}{letter}{num}"
            
            text = re.sub(UNIT_PATTERN, normalize_unit, text, flags=re.IGNORECASE)
            if text != original_text:  # Only log if changed
                print(f"   (Normalized units: {original_text} → {text})")

            lower_text = text.lower()
            blocked = False
            for phrase in FULL_BLOCK_PHRASES:
                if phrase.lower() in lower_text:
                    print(f"   (Blocked hallucination containing '{phrase}': {text})")
                    blocked = True
                    break
            if blocked:
                transcription_queue.task_done()
                continue

            for phrase in CUTOFF_PHRASES:
                idx = lower_text.find(phrase.lower())
                if idx != -1:
                    text = text[:idx].strip()
                    print(f"   (Truncated hallucinated phrase '{phrase}': {original_text})")
                    break

            # Quick & dirty repetition buster (add after all other cleanups)
            words = text.split()
            if len(words) > 6:
                for i in range(len(words)-3):
                   if words[i:i+3] == words[i+3:i+6]:
                        text = ' '.join(words[:i+3]) + " [repeated]"
                        break

            # Capitalization for clarity
            keywords = r'(dispatch|central|station 52|baystate|wing|cooley|amr|bravo \d+|engine \d+|ladder \d+|a\d+|e\d+|b\d+|bs\d+|bl\d+)'
            text = re.sub(keywords, lambda m: m.group(0).title(), text, flags=re.I)

            if text:
                output = f"[{timestamp}] ({duration:.1f}s) {text}"
                print(output)
                with _log_lock:
                    log_path = LOG_FILE
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(output + "\n")
                # Publish to MQTT (mirrors what is written to the log)
                if mqtt_pub:
                    try:
                        mqtt_pub.publish_transcript(
                            feed=FEED_DESCRIPTION,
                            text=text,
                            duration_s=duration,
                            hhmmss=timestamp,
                        )
                    except Exception as e:
                        print(f" (MQTT publish failed: {e})")
            else:
                print(f"   (Empty after cleanup — discarded)")

            transcription_queue.task_done()

        except Exception as e:
            print(f"Error in transcriber: {e}")

def process_audio():
    global last_activity_time, LOG_FILE, CURRENT_LOG_DATE, filter_state
    ffmpeg_process = get_ffmpeg_stream(STREAM_URL)
    
    worker = threading.Thread(target=transcriber_worker, args=(model, device))
    worker.daemon = True
    worker.start()
    
    audio_buffer = []
    is_recording = False
    silence_counter = 0
    silence_limit_chunks = int(SILENCE_LIMIT * SAMPLE_RATE * 2 / CHUNK_BYTES)
    chunk_count = 0
    retry_count = 0

    try:
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1).lower()
                if key == 'q':
                    print("\nQuit requested — cleaning up...")
                    raise KeyboardInterrupt

            ready, _, _ = select.select([ffmpeg_process.stdout], [], [], 0.5)
            if ready:
                raw_bytes = ffmpeg_process.stdout.read(CHUNK_BYTES)
            else:
                if ffmpeg_process.poll() is not None:
                    print("ffmpeg process died. Restarting stream...")
                    time.sleep(min(2 ** retry_count, 60))
                    ffmpeg_process.kill()
                    ffmpeg_process.wait()
                    ffmpeg_process = get_ffmpeg_stream(STREAM_URL)
                    retry_count += 1
                continue

            if not raw_bytes:
                print("Stream lost (EOF). Reconnecting...")
                time.sleep(min(2 ** retry_count, 60))
                ffmpeg_process.kill()
                ffmpeg_process.wait()
                ffmpeg_process = get_ffmpeg_stream(STREAM_URL)
                filter_state = np.zeros((sos.shape[0], 2))
                retry_count += 1
                continue

            retry_count = 0
            audio_chunk = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            audio_chunk, filter_state = signal.sosfilt(sos, audio_chunk, zi=filter_state)
            audio_chunk = audio_chunk.astype(np.float32)

            # WebRTC VAD
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            is_speech = False
            for i in range(0, len(audio_int16), VAD_FRAME_BYTES // 2):
                frame = audio_int16[i:i + VAD_FRAME_BYTES // 2].tobytes()
                if len(frame) == VAD_FRAME_BYTES:
                    if vad.is_speech(frame, SAMPLE_RATE):
                        is_speech = True
                        break

            current_time = time.time()
            current_date = datetime.date.today()
            
            if current_date != CURRENT_LOG_DATE:
                rollover_timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"[{rollover_timestamp}] [ROLLOVER] Day ended, continuing in new log\n")
                new_log = os.path.join(OUTPUT_FOLDER, f"{BASE_LOG_FILENAME}_{current_date}.log")
                os.makedirs(OUTPUT_FOLDER, exist_ok=True)
                if not os.path.exists(new_log):
                    open(new_log, 'a').close()
                with open(new_log, "a", encoding="utf-8") as f:
                    f.write(f"[{rollover_timestamp}] [STARTED] Transcription session continued - {FEED_DESCRIPTION} feed\n")
                with _log_lock:
                    LOG_FILE = new_log
                    CURRENT_LOG_DATE = current_date
                print(f"   [Rollover] Switched to new log: {new_log}")

            if current_time - last_activity_time > IDLE_THRESHOLD_SECONDS:
                last_heard = datetime.datetime.fromtimestamp(last_activity_time).strftime("%H:%M:%S")
                print(f"   [Idle >10 min] Last heard at {last_heard}")
                last_activity_time = current_time

            if is_speech:
                if not is_recording:
                    print("Voice started...")
                    is_recording = True
                audio_buffer.append(audio_chunk)
                silence_counter = 0
                last_activity_time = current_time
            else:
                if is_recording:
                    audio_buffer.append(audio_chunk)
                    silence_counter += 1

                    if silence_counter >= silence_limit_chunks:
                        full_audio = np.concatenate(audio_buffer)
                        
                        if len(full_audio) > 0:
                            percentile_val = np.percentile(np.abs(full_audio), NORMALIZATION_PERCENTILE)
                            if percentile_val > 0:
                                full_audio = full_audio / percentile_val
                                full_audio = np.clip(full_audio, -1.0, 1.0)
                        
                        full_audio = full_audio.astype(np.float32)

                        duration = len(full_audio) / SAMPLE_RATE
                        
                        if duration >= MIN_SPEECH_SECONDS:
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            transcription_queue.put((timestamp, full_audio.copy()))
                            print(f"   Queued {duration:.1f}s segment for transcription")
                        else:
                            print(f"   (Skipping short burst: {duration:.1f}s)")
                        
                        audio_buffer = []
                        is_recording = False
                        silence_counter = 0

            chunk_count += 1
            if chunk_count % GC_INTERVAL == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("\nStopping transcriptor...")
    finally:
        transcription_queue.put((None, None))
        worker.join()
        
        stop_timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{stop_timestamp}] [STOPPED] Transcription session ended")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{stop_timestamp}] [STOPPED] Transcription session ended\n")

        # Clean MQTT shutdown (will publish offline if availability is enabled)
        if mqtt_pub:
            try:
                mqtt_pub.close()
            except Exception:
                pass
        
        ffmpeg_process.kill()
        ffmpeg_process.wait()

if __name__ == "__main__":
    process_audio()
