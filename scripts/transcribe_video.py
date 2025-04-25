import os
import whisper
import json

def transcribe(input_video="data/raw/dopamine_detox.mp4", model_name="large"):
    os.makedirs("data/transcripts", exist_ok=True)

    print(f"🧠 Loading model '{model_name}' on CUDA (fp16)...")
    model = whisper.load_model(model_name, device="cuda")  # Uses GPU

    print(f"📼 Transcribing {input_video} ...")
    result = model.transcribe(input_video, language="en", verbose=True, fp16=True)  # Full accuracy + speed

    # File paths
    txt_path = "data/transcripts/dopamine_detox.txt"
    json_path = "data/transcripts/dopamine_detox.json"
    vtt_path = "data/transcripts/dopamine_detox.vtt"

    # Save .txt
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    # Save .json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, indent=2)

    # Save .vtt (subtitles)
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for segment in result["segments"]:
            start = format_time(segment["start"])
            end = format_time(segment["end"])
            text = segment["text"].strip()
            f.write(f"{start} --> {end}\n{text}\n\n")

    print("✅ Transcription complete. Files saved to /data/transcripts")

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")

if __name__ == "__main__":
    transcribe()
