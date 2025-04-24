import json
import subprocess
import re

def call_local_mistral(prompt: str) -> list[dict]:
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        response_text = result.stdout.decode("utf-8").strip()

        # Try direct load first
        try:
            return json.loads(response_text)

        except json.JSONDecodeError:
            # Fallback: extract first valid JSON array
            match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))

            print("❌ Model output contained no valid JSON list.")
            print("📝 Raw output:\n", response_text)
            return []

    except subprocess.CalledProcessError as e:
        print("❌ Ollama call failed:", e.stderr.decode("utf-8"))
        return []

