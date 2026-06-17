import os
import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_print(text):
    """Sanitizes text for Windows terminal output."""
    try:
        print(text.encode("ascii", errors="replace").decode("ascii"))
    except:
        print(text)

def generate_report(summary, output_dir=".", detected_events=None):
    """
    Prints a formatted report to console and saves to a .txt file.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"report_{timestamp}.txt"
    report_path = os.path.join(output_dir, report_filename)
    
    report_lines = []
    
    divider = "=" * 60
    report_lines.append(divider)
    report_lines.append("YOUTUBE SPORTS COMMENTARY ANALYSIS REPORT")
    report_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(divider)
    
    report_lines.append("\n[1] MATCH OVERVIEW")
    report_lines.append(divider)
    report_lines.append(f"Total Sentences Analyzed: {summary['total_sentences']}")
    report_lines.append(f"Total Sports Events Detected: {summary['total_events']}")
    report_lines.append(f"\nNarrative Summary:\n{summary['narrative']}")
    
    report_lines.append("\n[2] EVENT FREQUENCY")
    report_lines.append(divider)
    report_lines.append(f"{'Event Type':<20} | {'Frequency':<10}")
    report_lines.append("-" * 35)
    for event, count in summary['frequency_table']:
        report_lines.append(f"{event:<20} | {count:<10}")
        
    report_lines.append("\n[3] KEY MOMENTS")
    report_lines.append(divider)
    for moment in summary['key_moments']:
        report_lines.append(f"* {moment['type']} (x{moment['count']}): \"{moment['sample']}\"")
        
    report_lines.append("\n[4] EVENT TIMELINE")
    report_lines.append(divider)
    report_lines.append(f"{'Time':<10} | {'Event Type':<20} | {'Commentary Snippet'}")
    report_lines.append("-" * 60)
    for entry in summary['timeline'][:50]: # Limit for display
        snippet = entry['commentary'][:50] + "..." if len(entry['commentary']) > 50 else entry['commentary']
        report_lines.append(f"{entry['time']:<10} | {entry['type']:<20} | {snippet}")
    
    if len(summary['timeline']) > 50:
        report_lines.append(f"... and {len(summary['timeline']) - 50} more events.")
        
    # ── Section 5: Auto-Evaluation Metrics (computed from live data) ──────────
    if detected_events:
        # Separate events by which tier detected them
        tier1_events   = [e for e in detected_events if e.get("confidence","") == "high (tier 1)"]
        tier15_events  = [e for e in detected_events if e.get("confidence","") == "medium (tier 1.5)"]
        tier175_events = [e for e in detected_events if e.get("confidence","") == "score (tier 1.75)"]
        tier2_events   = [e for e in detected_events if "tier 2" in e.get("confidence", "")]
        total          = len(detected_events)

        # Parse confidence percentages from tier 2 strings e.g. "ml (78%) tier 2"
        tier2_confidences = []
        for e in tier2_events:
            try:
                pct_str = e["confidence"].split("(")[1].split("%")[0]
                tier2_confidences.append(float(pct_str))
            except Exception:
                pass

        avg_conf        = (sum(tier2_confidences) / len(tier2_confidences)) if tier2_confidences else 0.0
        total_sentences = summary.get("total_sentences", 0)
        rejected        = total_sentences - total

        def pct(n):
            return f"{(n/total*100):.1f}%" if total else "N/A"

        report_lines.append("\n[5] EVALUATION METRICS (Auto-Computed from This Video)")
        report_lines.append(divider)
        report_lines.append(f"{'Total Sentences Analyzed':<38}: {total_sentences}")
        report_lines.append(f"{'Total Events Detected':<38}: {total}")
        report_lines.append(f"{'Overall Detection Rate':<38}: {(total/total_sentences*100):.1f}% of sentences classified as events" if total_sentences else "")
        report_lines.append("")
        report_lines.append("-- Tier Contribution (Ablation Study) --")
        report_lines.append(f"{'Tier 1  (Keyword Phrases)':<38}: {len(tier1_events):>3} events  ({pct(len(tier1_events))} of detections)")
        report_lines.append(f"{'Tier 1.5 (Single-Word Regex)':<38}: {len(tier15_events):>3} events  ({pct(len(tier15_events))} of detections)")
        report_lines.append(f"{'Tier 1.75 (Numeric Scorelines)':<38}: {len(tier175_events):>3} events  ({pct(len(tier175_events))} of detections)")
        report_lines.append(f"{'Tier 2  (TF-IDF Label Spreading)':<38}: {len(tier2_events):>3} events  ({pct(len(tier2_events))} of detections)")
        report_lines.append("")
        report_lines.append("-- Confidence Metrics (Tier 2 ML) --")
        if tier2_confidences:
            report_lines.append(f"{'Average ML Confidence':<38}: {avg_conf:.1f}%")
            report_lines.append(f"{'Highest Confidence':<38}: {max(tier2_confidences):.1f}%")
            report_lines.append(f"{'Lowest Accepted Confidence':<38}: {min(tier2_confidences):.1f}%  (threshold: 65%)")
        else:
            report_lines.append("  No Tier 2 ML detections in this video.")
        report_lines.append("")
        report_lines.append("-- False Positive Suppression Proxy --")
        report_lines.append(f"{'Sentences Filtered as NEUTRAL':<38}: {rejected} sentences auto-rejected")
        report_lines.append(f"  (Tier 2 confidence below 65% or no rule/regex/score match)")
        report_lines.append(f"  Active suppression rate: {(rejected/total_sentences*100):.1f}% of input sentences" if total_sentences else "")

    report_text = "\n".join(report_lines)

    
    # Console Output
    for line in report_lines:
        safe_print(line)
        
    # File Output
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info(f"Report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save report file: {e}")
        
    return report_path
