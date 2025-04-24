import os
import json
from faster_whisper import WhisperModel

def transcribe(input_video="data/raw/dopamine_detox.mp4", model_size="large-v2"):
    # Setup
    os.makedirs("data/transcripts", exist_ok=True)
    txt_path = "data/transcripts/dopamine_detox.txt"
    json_path = "data/transcripts/dopamine_detox.json"
    vtt_path = "data/transcripts/dopamine_detox.vtt"

    # Load model on GPU with float16 precision
    print(f"🧠 Loading Faster-Whisper '{model_size}' on CUDA (float16)...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # Transcribe
    print(f"📼 Transcribing: {input_video}")
    segments, info = model.transcribe(input_video, beam_size=5, language="en")

    print(f"🌐 Detected language: {info.language}")
    
    all_segments = []
    with open(txt_path, "w", encoding="utf-8") as f_txt, \
         open(vtt_path, "w", encoding="utf-8") as f_vtt:
        
        f_vtt.write("WEBVTT\n\n")
        
        for i, segment in enumerate(segments):
            start = segment.start
            end = segment.end
            text = segment.text.strip()

            # Save each line to .txt
            f_txt.write(text + "\n")

            # Save each segment to list
            all_segments.append({
                "start": float(start),
                "end": float(end),
                "text": text
            })

            # Write to VTT
            f_vtt.write(f"{format_time(start)} --> {format_time(end)}\n{text}\n\n")

    # Save as JSON
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(all_segments, f_json, indent=2)

    print("✅ Transcription complete. Saved to: /data/transcripts")

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")

if __name__ == "__main__":
    transcribe()
