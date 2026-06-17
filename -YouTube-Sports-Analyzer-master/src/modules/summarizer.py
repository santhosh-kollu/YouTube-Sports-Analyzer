import logging
from datetime import timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_summary(detection_results, processed_data):
    """
    Aggregates detected events into a structured summary.
    """
    detected_events = detection_results["detected_events"]
    event_counts = detection_results["event_counts"]
    total_sentences = detection_results["total_sentences"]
    
    # 1. Timeline (formatted mm:ss)
    timeline = []
    for event in detected_events:
        start_time = event["timestamp"]["start"]
        time_str = str(timedelta(seconds=int(start_time)))[2:] # HH:MM:SS -> MM:SS
        timeline.append({
            "time": f"{time_str} ({int(start_time)}s)",
            "type": event["event"],
            "commentary": event["timestamp"]["text"]
        })
        
    # 2. Event Frequency Table (sorted)
    sorted_counts = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)
    frequency_table = [count for count in sorted_counts if count[0] != "NEUTRAL"]
    
    # 3. Key Moments (Top 3 event types)
    key_moments = []
    for event_type, count in frequency_table[:3]:
        if count > 0:
            # Find a representative sentence for this event
            sample = ""
            for e in detected_events:
                if e["event"] == event_type:
                    sample = e["timestamp"]["text"]
                    break
            key_moments.append({
                "type": event_type,
                "count": count,
                "sample": sample
            })
            
    # 4. Narrative Summary
    # Extract goals, fouls, and saves specifically for a natural feel
    goals = event_counts.get("GOAL", 0)
    fouls = event_counts.get("FOUL", 0)
    saves = event_counts.get("GOALKEEPER_SAVE", 0)
    cards = event_counts.get("YELLOW_CARD", 0) + event_counts.get("RED_CARD", 0)
    
    narrative = f"The match featured {goals} goals, making it an exciting affair. "
    narrative += f"There were {fouls} fouls committed, reflecting a physical contest. "
    narrative += f"The goalkeeper was kept busy with {saves} notable saves. "
    narrative += f"{cards} disciplinary cards were shown throughout the match."
    
    logger.info("Summary generation complete.")
    return {
        "timeline": timeline,
        "frequency_table": frequency_table,
        "key_moments": key_moments,
        "narrative": narrative,
        "total_events": len(detected_events),
        "total_sentences": total_sentences
    }
