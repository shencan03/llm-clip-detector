# scripts/detect_clips.py

import subprocess
import argparse
import json
from pathlib import Path

def load_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8")

def load_transcript(transcript_path: Path) -> str:
    return transcript_path.read_text(encoding="utf-8")

def run_ollama(prompt: str, model: str = "mistral") -> str:
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout

def extract_json_from_response(output: str) -> list:
    try:
        json_str = output[output.index("[") : output.rindex("]") + 1]
        return json.loads(json_str)
    except Exception as e:
        print("[!] Failed to parse JSON:", e)
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", type=str, required=True, help="Path to the transcript .txt file")
    parser.add_argument("--prompt", type=str, default="prompts/detect_clips_prompt.txt", help="Path to prompt template")
    parser.add_argument("--output", type=str, default="data/clips/detected_clips.json", help="Where to save the clips")
    args = parser.parse_args()

    prompt_template = load_prompt(Path(args.prompt))
    transcript = load_transcript(Path(args.transcript))

    combined_prompt = f"{prompt_template.strip()}\n\nTranscript:\n{transcript.strip()}"
    print("[*] Running Ollama inference...")

    raw_output = run_ollama(combined_prompt)
    clips = extract_json_from_response(raw_output)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(clips, f, indent=2)

    print(f"[+] Saved {len(clips)} clips to {args.output}")

if __name__ == "__main__":
    main()
