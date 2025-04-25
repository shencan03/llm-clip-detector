import os
import json
import subprocess

CHUNKS_DIR = "data/chunks"
OUTPUT_PATH = "data/detected_clips.json"

PROMPT_TEMPLATE = """
You are given a segment from a podcast transcript. Your task is to find up to two compact, complete dialogue sequences that:

- Start at a natural point in the conversation (not mid-sentence)
- Develop a specific idea, story, or point
- End with a clear conclusion, takeaway, or insight

Avoid including any part that feels like it’s cut off, incomplete, or starts in the middle of a thought. Output should follow this structure:

[
  {{
    "start_time": [float, in seconds],
    "end_time": [float, in seconds],
    "title": "[short summary or main point]"
  }}
]
Transcript:
"""

def call_mistral(prompt: str) -> list[dict]:
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        output = result.stdout.decode("utf-8").strip()

        try:
            start = output.find("[")
            end = output.rfind("]") + 1
            return json.loads(output[start:end])
        except Exception:
            print("❌ Failed to parse JSON from output. Output was:")
            print(output)
            return []
    except subprocess.CalledProcessError as e:
        print("❌ Ollama call failed:", e.stderr.decode("utf-8"))
        return []

def main():
    all_clips = []
    for filename in sorted(os.listdir(CHUNKS_DIR)):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(CHUNKS_DIR, filename), "r", encoding="utf-8") as f:
            chunk = json.load(f)

        full_text = " ".join(seg["text"] for seg in chunk)
        prompt = PROMPT_TEMPLATE + full_text
        clips = call_mistral(prompt)

        print(f"✅ Processed {filename}: {len(clips)} clips")
        all_clips.extend(clips)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_clips, f, indent=2)
    print(f"✅ Saved all detected clips to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()