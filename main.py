import sounddevice as sd
import numpy as np
import librosa
import math

# -----------------------------
# Hz → Note
# -----------------------------
def hz_to_note(frequency):
    if frequency <= 0:
        return None

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

    midi = round(69 + 12 * math.log2(frequency / 440.0))
    note = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1

    return f"{note}{octave}"

# -----------------------------
# CONFIG
# -----------------------------
sr = 22050
block_size = 2048  # chunk size
threshold = 0.02

# State
is_playing = False
current_segment = []

# -----------------------------
# Audio callback
# -----------------------------
def callback(indata, frames, time, status):
    global is_playing, current_segment

    y = indata[:, 0]  # mono

    # -------------------------
    # RMS (energy)
    # -------------------------
    rms = np.sqrt(np.mean(y**2))

    if rms > threshold:
        # SOUND detected
        current_segment.extend(y)

        if not is_playing:
            print("▶ START")
            is_playing = True

    else:
        # SILENCE detected
        if is_playing:
            print("⏸ END")

            # Process segment
            process_segment(np.array(current_segment))

            current_segment = []
            is_playing = False

# -----------------------------
# Process one segment
# -----------------------------
def process_segment(segment):

    if len(segment) < 2048:
        return

    # Pitch detection
    f0 = librosa.yin(segment, fmin=100, fmax=1000)
    f0 = f0[~np.isnan(f0)]

    if len(f0) == 0:
        return

    freq = np.median(f0)
    note = hz_to_note(freq)

    print(f"🎵 NOTE: {note} ({freq:.2f} Hz)")

# -----------------------------
# Start stream
# -----------------------------
with sd.InputStream(callback=callback,
                    channels=1,
                    samplerate=sr,
                    blocksize=block_size):

    print("Listening... Press Ctrl+C to stop.")
    while True:
        pass