import flask
from flask import Flask, render_template, request, Response, jsonify
import json
import time
import threading
import queue
import logging
import datetime
import os
import uuid
from src.modules.downloader import download_audio
from src.modules.transcriber import transcribe_audio
from src.modules.preprocessor import preprocess_transcript
from src.modules.event_detector import detect_events
from src.modules.summarizer import generate_summary

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary to store per-session progress queues (Fixes Global State Leakage)
progress_queues = {}

# Active jobs counter to prevent DoS via OOM
active_jobs = 0
MAX_JOBS = 2

def run_analysis_pipeline(url, model_size, session_id):
    """
    Background worker that runs the full pipeline and pushes status updates to SSE queue.
    """
    global active_jobs
    q = progress_queues.get(session_id)
    if not q:
        return

    try:
        # Step 1: Download
        q.put({"step": "download", "status": "active", "msg": "Downloading audio from YouTube..."})
        audio_output_path = f"data/audio_{session_id}.wav"
        
        audio_path = download_audio(url, output_path=audio_output_path)
        if not audio_path:
            raise Exception("Download failed. Could be restricted, a live stream, or over the 20-minute limit.")
        q.put({"step": "download", "status": "done", "msg": "Audio downloaded successfully."})
        
        # Step 2: Transcribe
        q.put({"step": "transcribe", "status": "active", "msg": "Transcribing audio (Whiser model: {})...".format(model_size)})
        whisper_result = transcribe_audio(audio_path, model_size)
        if not whisper_result:
            raise Exception("Transcription failed.")
        q.put({"step": "transcribe", "status": "done", "msg": "Transcription complete."})
        
        # Step 3: Preprocess
        q.put({"step": "preprocess", "status": "active", "msg": "Cleaning and splitting transcript..."})
        processed_data = preprocess_transcript(whisper_result)
        q.put({"step": "preprocess", "status": "done", "msg": "Preprocessing complete."})
        
        # Step 4: Detect
        q.put({"step": "detect", "status": "active", "msg": "Detecting sports events with Tier 1/2 models..."})
        detection_results = detect_events(processed_data)
        q.put({"step": "detect", "status": "done", "msg": "Event detection complete."})
        
        # Step 5: Summarize
        q.put({"step": "summarize", "status": "active", "msg": "Aggregating results and generating summary..."})
        summary = generate_summary(detection_results, processed_data)
        q.put({"step": "summarize", "status": "done", "msg": "Summary generated successfully.", "result": summary, "raw_text": whisper_result.get("text", ""), "detected_events": detection_results["detected_events"]})
        
    except Exception as e:
        logger.error(f"Pipeline error for session {session_id}: {e}")
        q.put({"step": "error", "status": "error", "msg": str(e)})
    finally:
        # Always decrement active jobs when thread finishes
        active_jobs -= 1
        
        # Securely delete the .wav file to prevent storage exhaustion
        audio_path_to_delete = f"data/audio_{session_id}.wav"
        if os.path.exists(audio_path_to_delete):
            try:
                os.remove(audio_path_to_delete)
                logger.info(f"Cleaned up audio file: {audio_path_to_delete}")
            except Exception as e:
                logger.error(f"Failed to delete {audio_path_to_delete}: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    global active_jobs
    url = request.form.get('url')
    model = request.form.get('model', 'tiny')
    
    # Input Validation
    if not url:
        return jsonify({"error": "No URL provided."}), 400
        
    if not (url.startswith("https://www.youtube.com/") or url.startswith("https://youtu.be/")):
        return jsonify({"error": "Invalid YouTube URL. Must start with https://www.youtube.com/ or https://youtu.be/"}), 400
    
    # DoS Protection: Check concurrent limits
    if active_jobs >= MAX_JOBS:
        return jsonify({"error": "Server is extremely busy right now (Max 2 concurrent jobs). Please try again in a few minutes!"}), 503
        
    active_jobs += 1
    session_id = str(uuid.uuid4())
    progress_queues[session_id] = queue.Queue()
    
    # Start pipeline in separate thread
    thread = threading.Thread(target=run_analysis_pipeline, args=(url, model, session_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started", "session_id": session_id})

@app.route('/stream')
def stream():
    session_id = request.args.get('session_id')
    if not session_id or session_id not in progress_queues:
        return Response("Invalid or missing session ID", status=400)
        
    q = progress_queues[session_id]
    
    def generate():
        while True:
            try:
                # Wait for next update in queue
                data = q.get()
                yield "data: {}\n\n".format(json.dumps(data))
                
                # Stop streaming after summarize is done or error occurred
                if data['step'] == 'summarize' and data['status'] == 'done':
                    break
                if data['step'] == 'error':
                    break
            except Exception as e:
                logger.error(f"SSE error for session {session_id}: {e}")
                break
                
        # Cleanup queue from memory after streaming completes
        if session_id in progress_queues:
            del progress_queues[session_id]
                
    return Response(generate(), mimetype='text/event-stream')
