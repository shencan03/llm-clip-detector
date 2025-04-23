# 🎬 LLM Clip Detector

This is an offline pipeline that identifies TikTok/YouTube Reels-style short, engaging clips from long podcast transcripts using:

- 🧠 Local LLM (Mistral-7B)
- 🎙️ Whisper for transcription
- ✂️ FFmpeg for video cutting

## 🧰 Features
- Detects hook–development–punchline structure
- Outputs clip start/end timestamps
- Runs 100% locally

## 💻 Setup
```bash
pip install -r requirements.txt
```

## 📂 Folder Structure
- `/scripts`: processing pipeline
- `/models`: GGUF LLM files (e.g., Mistral)
- `/data`: audio, transcripts, clips
