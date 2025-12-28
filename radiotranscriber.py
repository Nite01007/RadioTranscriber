import subprocess
import numpy as np
import whisper
import torch
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

from config import USERNAME, PASSWORD, FEED_NUMBER, FEED_DESCRIPTION

# --- SETTINGS ---
STREAM_URL = f"http://{USERNAME}:{PASSWORD}@audio.broadcastify.com/{FEED_NUMBER}.mp3"
MODEL_SIZE = "large-v3"
SAMPLE_RATE = 16000
CHUNK_BYTES = 8192
MIN_SPEECH_SECONDS = 1.0
SILENCE_LIMIT = 2.0
IDLE_THRESHOLD_SECONDS = 600
NORMALIZATION_PERCENTILE = 95
GC_INTERVAL = 100
NO_SPEECH_THRESHOLD = 0.8

BASE_LOG_FILENAME = f"transcription_{FEED_DESCRIPTION}"
LOG_FILE = f"{BASE_LOG_FILENAME}_{datetime.date.today()}.log"
CURRENT_LOG_DATE = datetime.date.today()

if not os.path.exists(LOG_FILE):
    open(LOG_FILE, 'a').close()

sos = signal.butter(5, 100 / (SAMPLE_RATE / 2), btype='high', output='sos')
filter_state = np.zeros((sos.shape[0], 2))

vad = webrtcvad.Vad(3)
VAD_FRAME_MS = 30
VAD_FRAME_BYTES = int(SAMPLE_RATE * (VAD_FRAME_MS / 1000) * 2)  # int16 bytes

INITIAL_PROMPT = (
    "Verbatim police, fire, EMS radio dispatch in English. Transcribe exactly as spoken: no paraphrasing, expansions, interpretations. "
    "No credits, captions like 'Transcription by', 'ESO', 'amara.org', 'Rev'. Speaker then recipient ID. Identifiers: Dispatch, Central, Station 52, units. "
    "'Station 52' not 'patient 52'. Squad 1 not squadron. Unit IDs: 52XX## (A Ambulance, E Engine, L Ladder, B Police, BS Sergeant, BL Lieutenant; e.g., 52A1, 52E1, 52B10). "
    "Shortened: 'A1', 'E1', 'Bravo 6', 'Engine 1'. Transcribe literally. Status: received, responding, en route, on scene, clear, in service, 10-4, copy. "
    "Transports: Baystate, Wing, Cooley. Addresses: street/numbers. Plates: phonetic (e.g., '1 Alpha Bravo Charlie 23'). Phonetic alphabet: NATO. AMR ambulance. "
    "Unit '52A2' is often spoken as 'five two eight two' or '52 82' — transcribe as '52A2'. "
    "Ignore non-speech, alert tones as '[beeps]', not 'bee', 'beep', or sustained letters. Speech: clipped, rapid, professional with static. No fillers, narration, music."
)

FULL_BLOCK_PHRASES = [
    "Transcription by", "CastingWords", "ESO", "Translation by",
    "Captions by", "Rev.com", "Rev", "Subtitle", "Subtitles",
    "amara.org", "amara", "subtitles by the amara.org community",
    "ESA, Inc.", "U.S. Department of", "Transcripts translated by",
    "Transcription Outsourcing", "Transcription Outsourcing, Inc"
]

CUTOFF_PHRASES = [
    "We'll be right back",
    "We will be right back",
    "Thank you for watching",
    "Thanks for watching",
    "Thank you for your patience",
    "Stay tuned",
    "Commercial break"
]

transcription_queue = queue.Queue()

# --- INITIALIZATION ---
print(f"Loading Whisper model '{MODEL_SIZE}'...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(MODEL_SIZE).to(device)
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
            
            result = model.transcribe(
                audio_data,
                language="en",
                fp16=(device == "cuda"),
                initial_prompt=INITIAL_PROMPT,
                condition_on_previous_text=False,
                temperature=0.0,
                beam_size=5,
                best_of=5,
                patience=1.5,
                suppress_blank=True
            )
            
            text = result['text'].strip()
            transcribe_time = time.time() - transcribe_start
            print(f"   Done in {transcribe_time:.1f}s")

            # No-speech prob filter
            if result.get('no_speech_prob', 0) > NO_SPEECH_THRESHOLD:
                print(f"   (Discarded non-speech segment: prob {result['no_speech_prob']:.2f})")
                transcription_queue.task_done()
                continue

            original_text = text

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

            # Map spoken numbers to letter units (your system)
            unit_map = {
                "eight two": "A2",
                "8 2": "A2",
                "82": "A2",
                "eight-two": "A2",
                "5 2 8 2": "52A2",
                "five two eight two": "52A2",
                "52 82": "52A2",
                "five-two-eight-two": "52A2"
            }
            lower_text = text.lower()
            for spoken, letter in unit_map.items():
                if spoken in lower_text:
                    text = re.sub(re.escape(spoken), letter, text, flags=re.IGNORECASE)
                    print(f"   (Mapped unit: {original_text} → {text})")

            lower_text = text.lower()  # Refresh
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

            # Capitalization for clarity
            keywords = r'(dispatch|central|station 52|baystate|wing|cooley|amr|bravo \d+|engine \d+|ladder \d+|a\d+|e\d+|b\d+|bs\d+|bl\d+)'
            text = re.sub(keywords, lambda m: m.group(0).title(), text, flags=re.I)

            if text:
                output = f"[{timestamp}] ({duration:.1f}s) {text}"
                print(output)
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(output + "\n")
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
                    ffmpeg_process.kill()
                    ffmpeg_process = get_ffmpeg_stream(STREAM_URL)
                continue

            if not raw_bytes:
                print("Stream lost (EOF). Reconnecting...")
                ffmpeg_process.kill()
                ffmpeg_process = get_ffmpeg_stream(STREAM_URL)
                filter_state = np.zeros((sos.shape[0], 2))
                continue

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
                LOG_FILE = f"{BASE_LOG_FILENAME}_{current_date}.log"
                if not os.path.exists(LOG_FILE):
                    open(LOG_FILE, 'a').close()
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"[{rollover_timestamp}] [STARTED] Transcription session continued - {FEED_DESCRIPTION} feed\n")
                CURRENT_LOG_DATE = current_date
                print(f"   [Rollover] Switched to new log: {LOG_FILE}")

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
        
        ffmpeg_process.kill()

if __name__ == "__main__":
    process_audio()
