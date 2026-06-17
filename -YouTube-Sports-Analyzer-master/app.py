import flask
from flask import Flask, render_template, request, Response, jsonify
import json
import time
import threading
import queue
import logging
import datetime
from modules.downloader import download_audio
from modules.transcriber import transcribe_audio
from modules.preprocessor import preprocess_transcript
from modules.event_detector import detect_events
from modules.summarizer import generate_summary

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Queue for SSE progress updates
progress_queue = queue.Queue()

def run_analysis_pipeline(url, model_size):
    """
    Background worker that runs the full pipeline and pushes status updates to SSE queue.
    """
    try:
        # Step 1: Download
        progress_queue.put({"step": "download", "status": "active", "msg": "Downloading audio from YouTube..."})
        audio_path = download_audio(url)
        if not audio_path:
            raise Exception("Download failed.")
        progress_queue.put({"step": "download", "status": "done", "msg": "Audio downloaded successfully."})
        
        # Step 2: Transcribe
        progress_queue.put({"step": "transcribe", "status": "active", "msg": "Transcribing audio (Whiser model: {})...".format(model_size)})
        whisper_result = transcribe_audio(audio_path, model_size)
        if not whisper_result:
            raise Exception("Transcription failed.")
        progress_queue.put({"step": "transcribe", "status": "done", "msg": "Transcription complete."})
        
        # Step 3: Preprocess
        progress_queue.put({"step": "preprocess", "status": "active", "msg": "Cleaning and splitting transcript..."})
        processed_data = preprocess_transcript(whisper_result)
        progress_queue.put({"step": "preprocess", "status": "done", "msg": "Preprocessing complete."})
        
        # Step 4: Detect
        progress_queue.put({"step": "detect", "status": "active", "msg": "Detecting sports events with Tier 1/2 models..."})
        detection_results = detect_events(processed_data)
        progress_queue.put({"step": "detect", "status": "done", "msg": "Event detection complete."})
        
        # Step 5: Summarize
        progress_queue.put({"step": "summarize", "status": "active", "msg": "Aggregating results and generating summary..."})
        summary = generate_summary(detection_results, processed_data)
        progress_queue.put({"step": "summarize", "status": "done", "msg": "Summary generated successfully.", "result": summary, "raw_text": whisper_result.get("text", ""), "detected_events": detection_results["detected_events"]})
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        progress_queue.put({"step": "error", "status": "error", "msg": str(e)})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    url = request.form.get('url')
    model = request.form.get('model', 'tiny')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    # Start pipeline in separate thread
    thread = threading.Thread(target=run_analysis_pipeline, args=(url, model))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started"})

@app.route('/stream')
def stream():
    def generate():
        while True:
            try:
                # Wait for next update in queue
                data = progress_queue.get()
                yield "data: {}\n\n".format(json.dumps(data))
                
                # Stop streaming after summarize is done or error occurred
                if data['step'] == 'summarize' and data['status'] == 'done':
                    break
                if data['step'] == 'error':
                    break
            except Exception as e:
                logger.error(f"SSE error: {e}")
                break
                
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
