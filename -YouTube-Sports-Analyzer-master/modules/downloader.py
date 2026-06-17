import os
import sys
import subprocess
import imageio_ffmpeg
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Browsers to try for cookie extraction (in order of preference)
BROWSERS = ["chrome", "edge", "firefox"]

def _build_command(youtube_url, output_path, ffmpeg_exe, browser=None):
    """Build the yt-dlp command, optionally with browser cookies."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-x",
        "--audio-format", "wav",
        "--ffmpeg-location", ffmpeg_exe,
        "--output", output_path.replace(".wav", ".%(ext)s"),
        "--no-playlist",
        "--socket-timeout", "30",
        "--retries", "3",
        "--add-header", "Accept-Language:en-US,en;q=0.9",
    ]
    if browser:
        cmd += ["--cookies-from-browser", browser]
    cmd.append(youtube_url)
    return cmd

def download_audio(youtube_url, output_path="audio_input.wav"):
    """
    Downloads audio from YouTube using yt-dlp and converts it to WAV.
    Automatically tries browser cookies (Chrome → Edge → Firefox) to
    bypass YouTube bot-detection and 429 rate-limiting errors.
    """
    logger.info(f"Starting download for URL: {youtube_url}")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Clean up old file
    if os.path.exists(output_path):
        os.remove(output_path)

    # Attempt 1 — no cookies (fastest, works on many videos)
    attempts = [None] + BROWSERS   # [no-cookie, chrome, edge, firefox]

    for attempt in attempts:
        label = f"cookies-from-browser={attempt}" if attempt else "no cookies"
        logger.info(f"Trying download: {label}")
        command = _build_command(youtube_url, output_path, ffmpeg_exe, browser=attempt)

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"Download successful ({label}). File: {output_path} ({size_mb:.2f} MB)")
                return output_path
            else:
                logger.warning(f"Output file not found after download attempt ({label}).")

        except subprocess.CalledProcessError as e:
            err = e.stderr or ""
            logger.warning(f"Download attempt failed ({label}): {err[-300:]}")
            # If rate-limited, wait before retrying
            if "429" in err or "Too Many Requests" in err:
                logger.info("Rate limited (429). Waiting 5 seconds before next attempt...")
                time.sleep(5)
            continue

        except Exception as e:
            logger.error(f"Unexpected error ({label}): {e}")
            continue

    logger.error("All download attempts failed. Please check your internet connection or try a different video URL.")
    return None

if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    download_audio(test_url)
