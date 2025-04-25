import json
import os

INPUT_PATH = "data/transcripts/dopamine_detox.json"
OUTPUT_DIR = "data/chunks"
MAX_TOKENS = 700  # adjust depending on prompt size + model limit

os.makedirs(OUTPUT_DIR, exist_ok=True)

def estimate_tokens(text):
    # 1 token ≈ 4 characters in English
    return len(text) // 4

def chunk_transcript():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        segments = json.load(f)

    chunks = []
    current_chunk = []
    current_text = ""

    for seg in segments:
        new_text = seg["text"].strip()
        if estimate_tokens(current_text + new_text) < MAX_TOKENS:
            current_chunk.append(seg)
            current_text += " " + new_text
        else:
            chunks.append(current_chunk)
            current_chunk = [seg]
            current_text = new_text

    if current_chunk:
        chunks.append(current_chunk)

    # Save chunks to individual files
    for i, chunk in enumerate(chunks):
        with open(f"{OUTPUT_DIR}/chunk_{i+1:03}.json", "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2)

    print(f"✅ Created {len(chunks)} chunks in /data/chunks")

if __name__ == "__main__":
    chunk_transcript()
