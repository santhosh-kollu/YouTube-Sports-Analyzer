import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_transcript(whisper_result):
    """
    Cleans transcript and splits into sentences with timestamps.
    """
    segments = whisper_result.get('segments', [])
    processed_data = {
        "sentences": [],
        "cleaned_sentences": [],
        "timestamps": [] # Each entry: {"start": sec, "end": sec, "text": original}
    }
    
    # Common English stopwords + sports fillers
    stopwords = set(["the", "is", "at", "which", "on", "and", "a", "of", "to", "in", "it", "that", "was", "for", "with", "as"])
    
    for seg in segments:
        text = seg['text'].strip()
        if not text:
            continue
            
        # Split by sentence delimiters
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for s in sentences:
            if not s.strip():
                continue
                
            # Clean sentence
            clean = s.lower()
            clean = re.sub(r'[^\w\s]', '', clean) # Remove punctuation
            
            # Remove stopwords
            words = clean.split()
            filtered_words = [w for w in words if w not in stopwords]
            clean_filtered = " ".join(filtered_words)
            
            processed_data["sentences"].append(s)
            processed_data["cleaned_sentences"].append(clean_filtered)
            processed_data["timestamps"].append({
                "start": seg['start'],
                "end": seg['end'],
                "text": s
            })
            
    logger.info(f"Preprocessing complete. Extracted {len(processed_data['sentences'])} sentences.")
    return processed_data
