import whisper
import numpy as np
import soundfile as sf
import librosa
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transcribe_audio(audio_path, model_size="base"):
    """
    Transcribes audio using OpenAI Whisper.
    Loads audio into memory to avoid ffmpeg subprocess issues on Windows.
    """
    logger.info(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    
    logger.info(f"Loading audio file: {audio_path}")
    try:
        # Load audio using soundfile
        audio_data, samplerate = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Resample to 16kHz if necessary (Whisper requirement)
        if samplerate != 16000:
            logger.info(f"Resampling audio from {samplerate}Hz to 16000Hz")
            audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
        
        # Ensure it's float32
        audio_data = audio_data.astype(np.float32)
        
        logger.info("Starting transcription...")
        result = model.transcribe(
            audio_data,
            verbose=False,
            temperature=0.0,                     # Fully deterministic — no random sampling
            condition_on_previous_text=False,    # Critical fix: prevents hallucination loops on crowd noise
            beam_size=5,                         # Better quality than greedy decoding
            best_of=5,                           # Pick best among 5 candidate transcriptions
            no_speech_threshold=0.6,             # Discard segments with >60% silence probability
            compression_ratio_threshold=2.4,     # Discard suspiciously repetitive output
            logprob_threshold=-1.0               # Discard low-confidence token predictions
        )
        
        logger.info("Transcription complete.")
        return result # result contains 'text' and 'segments'
        
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return None

if __name__ == "__main__":
    # Test
    if os.path.exists("audio_input.wav"):
        res = transcribe_audio("audio_input.wav", "tiny")
        if res:
            print(res['text'][:500])
