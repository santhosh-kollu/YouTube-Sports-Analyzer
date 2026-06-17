import argparse
import sys
import os
import logging
from modules.downloader import download_audio
from modules.transcriber import transcribe_audio
from modules.preprocessor import preprocess_transcript
from modules.event_detector import detect_events
from modules.summarizer import generate_summary
from modules.reporter import generate_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline(youtube_url, model_size="base", output_dir=".", demo_mode=False):
    """
    Main orchestration function for the YouTube Sports Commentary Analyzer pipeline.
    """
    logger.info("=" * 40)
    logger.info("YOUTUBE SPORTS COMMENTARY ANALYZER")
    logger.info("=" * 40)
    
    if demo_mode:
        logger.info("Demo Mode (Simulator) Activated...")
        # Simulating results for demonstration
        whisper_result = {
            "text": "The game kicks off now. What a goal! He scores from distance. A foul in the middle. Corner kick coming in.",
            "segments": [
                {"start": 0, "end": 5, "text": "The game kicks off now."},
                {"start": 10, "end": 15, "text": "What a goal! He scores from distance."},
                {"start": 30, "end": 35, "text": "A foul in the middle."},
                {"start": 60, "end": 65, "text": "Corner kick coming in."}
            ]
        }
        audio_path = "demo_audio.wav"
    else:
        # Step 1: Download
        audio_path = download_audio(youtube_url)
        if not audio_path:
            logger.error("Download failed. Pipeline aborted.")
            return None
            
        # Step 2: Transcribe
        whisper_result = transcribe_audio(audio_path, model_size)
        if not whisper_result:
            logger.error("Transcription failed. Pipeline aborted.")
            return None
            
    # Step 3: Preprocess
    processed_data = preprocess_transcript(whisper_result)
    
    # Step 4: Detect Events
    detection_results = detect_events(processed_data)
    
    # Step 5: Summarize
    summary = generate_summary(detection_results, processed_data)
    
    # Step 6: Report
    report_path = generate_report(summary, output_dir, detected_events=detection_results["detected_events"])
    
    logger.info("Pipeline Execution Successful!")
    return {
        "summary": summary,
        "report_path": report_path,
        "raw_text": whisper_result.get("text", "")
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Sports Commentary Analyzer")
    parser.add_argument("--url", help="YouTube video URL")
    parser.add_argument("--model", default="tiny", help="Whisper model size (tiny/base/small/medium)")
    parser.add_argument("--output", default=".", help="Output directory for the report .txt file")
    parser.add_argument("--demo", action="store_true", help="Runs with built-in sample commentary (no internet required)")
    
    args = parser.parse_args()
    
    if not args.url and not args.demo:
        parser.print_help()
        sys.exit(1)
        
    run_pipeline(args.url, args.model, args.output, args.demo)
