import subprocess
import os

def download(video_url: str, output_path: str = "data/raw/dopamine_detox.mp4"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = f'yt-dlp -f "bestvideo[height<=1080]+bestaudio/best" --merge-output-format mp4 -o "{output_path}" "{video_url}"'

    subprocess.run(command, shell=True, check=True)
    print(f"✅ Downloaded video to {output_path}")

if __name__ == "__main__":
    download("https://www.youtube.com/watch?v=p9JOpO5JvU0")
