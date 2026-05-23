from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np

# Load once at startup
print("Loading AI Model...")
model = WhisperModel("tiny", device="cpu", compute_type="float32")

def record_and_transcribe(seconds=5, samplerate=16000):
    print(f"\nRecording for {seconds} seconds...")
    # Records into a numpy array directly
    audio_data = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    
    # Process
    audio_array = np.squeeze(audio_data)
    segments, info = model.transcribe(audio_array, beam_size=5)
    
    return " ".join([segment.text.strip() for segment in segments])

if __name__ == "__main__":
    result = record_and_transcribe(seconds=5)
    print(f"\n--- TRANSLATED TEXT ---\n{result}")