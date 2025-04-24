import os
from nltk.tokenize import sent_tokenize
import tiktoken

def chunk_transcript(file_path="data/transcripts/dopamine_detox.txt", out_dir="data/chunks", max_tokens=4500):
    os.makedirs(out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("cl100k_base")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = sent_tokenize(text)
    chunks, current = [], ""

    for sentence in sentences:
        if len(enc.encode(current + sentence)) <= max_tokens:
            current += sentence + " "
        else:
            chunks.append(current.strip())
            current = sentence + " "
    if current:
        chunks.append(current.strip())

    for i, chunk in enumerate(chunks):
        with open(f"{out_dir}/chunk_{i+1}.txt", "w", encoding="utf-8") as f:
            f.write(chunk)

    print(f"✅ Created {len(chunks)} chunks in {out_dir}/")

if __name__ == "__main__":
    chunk_transcript()
