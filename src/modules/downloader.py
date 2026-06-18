import os
import sys
import subprocess
import imageio_ffmpeg
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _build_command(youtube_url, output_path, ffmpeg_exe):
    """Build the yt-dlp command with aggressive anti-bot spoofing."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-x",
        "--audio-format", "wav",
        "--ffmpeg-location", ffmpeg_exe,
        "--output", output_path.replace(".wav", ".%(ext)s"),
        "--no-playlist",
        "--match-filter", "duration <= 1200",
        "--socket-timeout", "30",
        "--retries", "5",
        
        # Anti-Bot / Anti-Throttle Bypasses:
        "--extractor-args", "youtube:player_client=android,ios,tv", # Spoof Android/iOS/TV clients
        "--impersonate", "chrome", # Use curl_cffi to forge a Google Chrome TLS Fingerprint
        "--legacy-server-connect", # Fix some SSL handshake errors
        "--no-check-certificate", # Ignore strict SSL checks if throttled
        "--add-header", "User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]
    
    cmd.append("--")
    cmd.append(youtube_url)
    return cmd

def download_audio(youtube_url, output_path="audio_input.wav"):
    """
    Downloads audio from YouTube using yt-dlp and converts it to WAV.
    """
    logger.info(f"Starting download for URL: {youtube_url}")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Clean up old file
    if os.path.exists(output_path):
        os.remove(output_path)

    command = _build_command(youtube_url, output_path, ffmpeg_exe)

    try:
        logger.info(f"Executing yt-dlp command to download audio...")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Download successful. File: {output_path} ({size_mb:.2f} MB)")
            return output_path
        else:
            logger.warning(f"Output file not found after download attempt.")

    except subprocess.CalledProcessError as e:
        err = e.stderr or str(e)
        logger.warning(f"Download attempt failed: {err[-1000:]}") # Keep some logs on backend
        raise Exception("Download failed. Could be restricted, a live stream, or over the 20-minute limit.")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise Exception("Download failed due to an unexpected error. Please try again.")

    raise Exception("Download failed. Could be restricted, a live stream, or over the 20-minute limit.")

if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    download_audio(test_url)
