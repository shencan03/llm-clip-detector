import os
import whisperx
import torch
import json

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("❌ Missing Hugging Face token. Please set HF_TOKEN in your .env file.")

# -------- SETTINGS -------- #
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TRANSCRIPT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "transcripts")
MODEL_SIZE = "large-v2"  # You have the power now (RTX 4080)

os.makedirs(TRANSCRIPT_OUTPUT_DIR, exist_ok=True)

def transcribe_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model(MODEL_SIZE, device)

    for filename in os.listdir(RAW_VIDEO_DIR):
        if filename.endswith(".mp4"):
            video_path = os.path.join(RAW_VIDEO_DIR, filename)
            basename = os.path.splitext(filename)[0]
            print(f"\n🔍 Transcribing: {filename}")

            json_out = os.path.join(TRANSCRIPT_OUTPUT_DIR, f"{basename}.json")
            txt_out = os.path.join(TRANSCRIPT_OUTPUT_DIR, f"{basename}.txt")

            if os.path.exists(json_out) and os.path.exists(txt_out):
                print(f"⚠️ Skipping {filename} — already transcribed.")
                continue

            # 1. Transcribe
            audio = whisperx.load_audio(video_path)
            result = model.transcribe(audio, batch_size=16)

            # 2. Word-level Alignment
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device)

            # 3. Diarization
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=HF_TOKEN,  # Replace this with your Hugging Face token
                device=device
            )
            diarize_segments = diarize_model(audio)

            # Combine diarization with segments
            result["segments"] = diarize_segments.to_dict(orient="records")

            # Clean up any unserializable fields
            for segment in result["segments"]:
                for key in list(segment):
                    if hasattr(segment[key], "__dict__"):
                        segment[key] = str(segment[key])

            # 4. Save as JSON
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # 5. Save readable .txt version
            with open(txt_out, "w", encoding="utf-8") as f:
                for seg in result["segments"]:
                    start = float(seg.get("start", 0))
                    end = float(seg.get("end", 0))
                    speaker = seg.get("speaker", "Unknown")
                    text = seg.get("text", "")
                    f.write(f"[{start:.2f} - {end:.2f}] ({speaker}) {text}\n")

            print(f"✅ Transcript saved: {txt_out} and {json_out}")

if __name__ == "__main__":
    transcribe_all()
