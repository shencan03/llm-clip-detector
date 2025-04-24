import subprocess
import os

def transcribe(input_video="data/raw/dopamine_detox.mp4", model="large", language="en"):
    os.makedirs("data/transcripts", exist_ok=True)
    command = [
        "whisper",
        input_video,
        "--model", model,
        "--output_format", "txt",
        "--language", language,
        "--output_dir", "data/transcripts"
    ]
    subprocess.run(command, check=True)
    print("✅ Transcription complete. Output in /data/transcripts")

if __name__ == "__main__":
    transcribe()
