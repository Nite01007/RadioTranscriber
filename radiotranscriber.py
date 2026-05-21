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
import http.server
import socketserver
# Optional MQTT publishing (for Home Assistant dashboard)
try:
    from mqtt_publisher import MqttPublisher
except Exception:
    MqttPublisher = None


# Load configuration
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
# Extract config values
SOURCE = config["source"]
if SOURCE == "broadcastify":
    USERNAME = config["broadcastify"]["username"]
    PASSWORD = config["broadcastify"]["password"]
    FEED_NUMBER = config["broadcastify"]["feed_number"]
else:
    USERNAME = PASSWORD = FEED_NUMBER = None
RTLSDR_CFG = config.get("rtlsdr", {})
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
if SOURCE == "broadcastify":
    STREAM_URL = f"http://{USERNAME}:{PASSWORD}@audio.broadcastify.com/{FEED_NUMBER}.mp3"
SAMPLE_RATE = 16000
CHUNK_BYTES = 8192
STREAM_PORT = int(RTLSDR_CFG["stream_port"]) if SOURCE == "rtlsdr" and "stream_port" in RTLSDR_CFG else None
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

if SOURCE == "broadcastify":
    redacted_url = f"http://{USERNAME}:********@audio.broadcastify.com/{FEED_NUMBER}.mp3"
    print(f"Streaming {FEED_DESCRIPTION} via Broadcastify feed {FEED_NUMBER}: {redacted_url}")
else:
    freqs = ", ".join(str(f) for f in RTLSDR_CFG["frequencies"])
    print(f"Streaming {FEED_DESCRIPTION} via RTL-SDR (device {RTLSDR_CFG['device_index']}) on {freqs}")
if SOURCE == "rtlsdr":
    print(f"   Press 'Q' to quit, '+'/'-' to raise/lower squelch (currently {RTLSDR_CFG['squelch']})")
else:
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

def get_rtlsdr_stream(frequencies, gain, squelch, device_index):
    freq_args = []
    for f in frequencies:
        freq_args += ['-f', str(f)]
    command = [
        'rtl_fm',
        '-d', str(device_index),
        *freq_args,
        '-M', 'fm', '-s', '200k', '-r', str(SAMPLE_RATE),
        '-l', str(squelch), '-g', str(gain),
        '-'
    ]
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

def _open_stream():
    if SOURCE == "broadcastify":
        return get_ffmpeg_stream(STREAM_URL)
    return get_rtlsdr_stream(
        RTLSDR_CFG["frequencies"],
        RTLSDR_CFG["gain"],
        RTLSDR_CFG["squelch"],
        RTLSDR_CFG["device_index"],
    )

def _write_squelch_to_config(value):
    try:
        with open("config.yaml", "r") as f:
            content = f.read()
        content = re.sub(
            r'^(\s*squelch:\s*)\d+',
            lambda m: m.group(1) + str(value),
            content,
            flags=re.MULTILINE,
        )
        with open("config.yaml", "w") as f:
            f.write(content)
    except Exception as e:
        print(f"   (Could not save squelch to config.yaml: {e})")


# --- HTTP audio streaming (rtlsdr only, when stream_port is configured) ---
_http_pcm_queue = queue.Queue(maxsize=50)
_stream_clients = []
_stream_clients_lock = threading.Lock()
_current_encoder = None
_current_encoder_lock = threading.Lock()


def _start_http_encoder():
    return subprocess.Popen(
        ['/usr/bin/ffmpeg',
         '-fflags', '+nobuffer',
         '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', '1', '-i', 'pipe:0',
         '-f', 'mp3', '-b:a', '32k', '-flush_packets', '1',
         '-loglevel', 'quiet', 'pipe:1'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,  # unbuffered stdin — writes go straight to the OS pipe
    )


def _http_encoder_feeder(proc, stop_event):
    FEED_BYTES = 512                          # ~16ms per write at 16000 Hz s16le mono
    feed_sec = FEED_BYTES / (SAMPLE_RATE * 2)
    silence = bytes(FEED_BYTES)
    buf = bytearray()
    next_feed = time.monotonic()

    while not stop_event.is_set():
        # Greedily drain all available PCM into the local buffer
        while True:
            try:
                chunk = _http_pcm_queue.get_nowait()
                if chunk is None:
                    return
                buf.extend(chunk)
            except queue.Empty:
                break

        # Sleep until the next scheduled write, keeping pace with real-time audio
        now = time.monotonic()
        if now < next_feed:
            time.sleep(next_feed - now)
        next_feed += feed_sec
        # If we fall badly behind (e.g. after a stall), reset rather than burst to catch up
        if next_feed < time.monotonic() - 0.5:
            next_feed = time.monotonic()

        # Write one 16ms piece; silence-fill when no real audio is queued
        if len(buf) >= FEED_BYTES:
            piece = bytes(buf[:FEED_BYTES])
            del buf[:FEED_BYTES]
        else:
            piece = silence

        try:
            proc.stdin.write(piece)
        except (OSError, BrokenPipeError):
            break

    try:
        proc.stdin.close()
    except OSError:
        pass


def _http_encoder_reader(proc, stop_event):
    while not stop_event.is_set():
        try:
            data = proc.stdout.read(4096)
        except OSError:
            break
        if not data:
            break
        with _stream_clients_lock:
            stale = []
            for client_q in _stream_clients:
                try:
                    client_q.put_nowait(data)
                except queue.Full:
                    stale.append(client_q)
            for client_q in stale:
                _stream_clients.remove(client_q)


class _StreamHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'audio/mpeg')
        self.send_header('Cache-Control', 'no-cache, no-store')
        self.end_headers()
        client_q = queue.Queue(maxsize=100)
        with _stream_clients_lock:
            _stream_clients.append(client_q)
            count = len(_stream_clients)
        print(f"   [Stream] Listener joined from {self.client_address[0]} — {count} listening")
        try:
            while True:
                try:
                    data = client_q.get(timeout=5.0)
                except queue.Empty:
                    continue
                if data is None:
                    break
                self.wfile.write(data)
                self.wfile.flush()
        except (OSError, BrokenPipeError):
            pass
        finally:
            with _stream_clients_lock:
                if client_q in _stream_clients:
                    _stream_clients.remove(client_q)
                count = len(_stream_clients)
            print(f"   [Stream] Listener left from {self.client_address[0]} — {count} listening")

    def log_message(self, *args):
        pass


def _http_monitor(port, stop_event):
    global _current_encoder
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer(('', port), _StreamHandler)
    server.daemon_threads = True
    svr_t = threading.Thread(target=server.serve_forever, daemon=True)
    svr_t.start()
    print(f"   [Stream] HTTP audio stream on :{port} — connect with http://<host>:{port}/")
    while not stop_event.is_set():
        while True:
            try:
                _http_pcm_queue.get_nowait()
            except queue.Empty:
                break
        proc = _start_http_encoder()
        with _current_encoder_lock:
            _current_encoder = proc
        feeder_t = threading.Thread(target=_http_encoder_feeder, args=(proc, stop_event), daemon=True)
        reader_t = threading.Thread(target=_http_encoder_reader, args=(proc, stop_event), daemon=True)
        feeder_t.start()
        reader_t.start()
        feeder_t.join()
        reader_t.join()
        proc.wait()
        with _current_encoder_lock:
            _current_encoder = None
        if not stop_event.is_set():
            print("   [Stream] Encoder restarting...")
            time.sleep(1)
    with _stream_clients_lock:
        for cq in _stream_clients:
            try:
                cq.put_nowait(None)
            except queue.Full:
                pass
    server.shutdown()


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
    ffmpeg_process = _open_stream()

    if STREAM_PORT:
        _http_stop = threading.Event()
        http_monitor_t = threading.Thread(target=_http_monitor, args=(STREAM_PORT, _http_stop), daemon=True)
        http_monitor_t.start()
    else:
        _http_stop = None
        http_monitor_t = None

    worker = threading.Thread(target=transcriber_worker, args=(model, device))
    worker.daemon = True
    worker.start()
    
    audio_buffer = []
    is_recording = False
    silence_counter = 0
    silence_limit_chunks = int(SILENCE_LIMIT * SAMPLE_RATE * 2 / CHUNK_BYTES)
    chunk_count = 0
    retry_count = 0
    pending_squelch = None

    try:
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1).lower()
                if key == 'q':
                    print("\nQuit requested — cleaning up...")
                    raise KeyboardInterrupt
                elif SOURCE == "rtlsdr" and key in ('+', '='):
                    pending_squelch = min(RTLSDR_CFG["squelch"] + 5, 100)
                    print(f"   [Squelch] {RTLSDR_CFG['squelch']} → {pending_squelch}")
                elif SOURCE == "rtlsdr" and key == '-':
                    pending_squelch = max(RTLSDR_CFG["squelch"] - 5, 0)
                    print(f"   [Squelch] {RTLSDR_CFG['squelch']} → {pending_squelch}")

            if pending_squelch is not None:
                RTLSDR_CFG["squelch"] = pending_squelch
                _write_squelch_to_config(pending_squelch)
                pending_squelch = None
                ffmpeg_process.kill()
                ffmpeg_process.wait()
                ffmpeg_process = _open_stream()
                filter_state = np.zeros((sos.shape[0], 2))
                retry_count = 0
                continue

            ready, _, _ = select.select([ffmpeg_process.stdout], [], [], CHUNK_BYTES / (SAMPLE_RATE * 2))
            if ready:
                raw_bytes = ffmpeg_process.stdout.read(CHUNK_BYTES)
            else:
                if ffmpeg_process.poll() is not None:
                    print("Stream process died. Restarting...")
                    time.sleep(min(2 ** retry_count, 60))
                    ffmpeg_process.kill()
                    ffmpeg_process.wait()
                    ffmpeg_process = _open_stream()
                    retry_count += 1
                    continue
                # rtl_fm is alive but not writing — squelch is closed between transmissions.
                # Synthesize silence so the VAD can accumulate silence frames and finalize
                # any open recording instead of stalling indefinitely.
                raw_bytes = bytes(CHUNK_BYTES)

            if not raw_bytes:
                print("Stream lost (EOF). Reconnecting...")
                time.sleep(min(2 ** retry_count, 60))
                ffmpeg_process.kill()
                ffmpeg_process.wait()
                ffmpeg_process = _open_stream()
                filter_state = np.zeros((sos.shape[0], 2))
                retry_count += 1
                continue

            retry_count = 0
            if STREAM_PORT:
                try:
                    _http_pcm_queue.put_nowait(raw_bytes)
                except queue.Full:
                    pass
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
        
        if STREAM_PORT and _http_stop:
            _http_stop.set()
            try:
                _http_pcm_queue.put_nowait(None)
            except queue.Full:
                pass
            with _current_encoder_lock:
                enc = _current_encoder
            if enc:
                try:
                    enc.kill()
                    enc.wait()
                except Exception:
                    pass
            if http_monitor_t:
                http_monitor_t.join(timeout=5)

        ffmpeg_process.kill()
        ffmpeg_process.wait()

if __name__ == "__main__":
    process_audio()
