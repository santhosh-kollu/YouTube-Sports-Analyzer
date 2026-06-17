# 🏟️ YouTube Sports Commentary Analyzer

A Flask-based web application that analyzes YouTube sports commentary videos using a multi-tier NLP pipeline. It downloads audio, transcribes it with OpenAI Whisper, and detects key match events (goals, fouls, cards, saves, and more) — all in real time via a live progress stream.

---

## 📸 Overview

Paste any YouTube sports video URL into the web interface, choose a Whisper model size, and the system automatically runs through a **6-stage pipeline** to identify and report on all significant match events.

---

## ✨ Features

- 🎥 **YouTube Audio Download** — Downloads audio from any public YouTube sports video using `yt-dlp`
- 🎙️ **Whisper Transcription** — Uses OpenAI Whisper with anti-hallucination parameters (deterministic, beam search, silence thresholding)
- 🧹 **Text Preprocessing** — Normalizes, cleans, and sentence-splits the transcript for structured analysis
- ⚽ **4-Tier Event Detection Engine** — Hybrid NLP approach combining rule-based and ML methods
- 📊 **Automated Evaluation Metrics** — Tier-contribution ablation, ML confidence scores, and neutral suppression rates
- 📡 **Real-Time Progress Streaming** — Live pipeline updates via Server-Sent Events (SSE)
- 📄 **Report Generation** — Saves a structured `.txt` report to disk after each analysis

---

## 🧠 Event Detection — 4-Tier Architecture

| Tier | Method | Confidence | Description |
|------|--------|------------|-------------|
| **Tier 1** | Multi-word keyword phrase matching | ⬆️ High | Matches curated multi-word phrases (e.g. `"back of the net"`, `"points to the spot"`) for 11 event types using stopword-cleaned text |
| **Tier 1.5** | Single-word regex with `\b` anchors | ⬆️ Medium-High | Fires on unclassified sentences only; uses word-boundary anchors to avoid false matches (`\bgoal\b` ≠ `goalkeeper`) |
| **Tier 1.75** | Numeric scoreline pattern matching | ➡️ Medium | Detects score announcements (`"2-1"`, `"two nil"`, `"makes it 3"`) as GOAL signals even without explicit keywords |
| **Tier 2** | TF-IDF + Semi-Supervised Label Spreading | ⬇️ Lower | Vectorizes unclassified sentences and propagates labels from hand-written seed examples; accepts predictions with ≥ 65% confidence only |

**Supported Event Types:** `GOAL`, `FOUL`, `PENALTY`, `YELLOW_CARD`, `RED_CARD`, `CORNER_KICK`, `OFFSIDE`, `GOALKEEPER_SAVE`, `SUBSTITUTION`, `FREE_KICK`, `NEAR_MISS`

> Deduplication removes repeated detections of the same event type within a **10-second sliding window**, preventing inflated counts from excited multi-sentence commentary.

---

## 🔄 Pipeline Stages

```
YouTube URL
    │
    ▼
[Stage 1] Download Audio       ──── yt-dlp + imageio-ffmpeg
    │
    ▼
[Stage 2] Transcribe Audio     ──── OpenAI Whisper (tiny / base / small / medium)
    │
    ▼
[Stage 3] Preprocess Text      ──── Normalize, split sentences, remove stopwords
    │
    ▼
[Stage 4] Detect Events        ──── 4-Tier Hybrid NLP Engine
    │
    ▼
[Stage 5] Summarize Results    ──── Frequency table, timeline, narrative, key moments
    │
    ▼
[Stage 6] Generate Report      ──── Console + .txt file output with evaluation metrics
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Web Framework | Flask |
| Audio Download | yt-dlp, imageio-ffmpeg |
| Transcription | OpenAI Whisper, PyTorch |
| Audio Processing | soundfile, librosa, numpy |
| NLP / ML | scikit-learn (TF-IDF, LabelSpreading) |
| Frontend | HTML, CSS, JavaScript (SSE) |

---

## 📁 Project Structure

```
YouTube-Sports-Analyzer/
│
├── app.py                        # Flask app: routes, SSE stream, pipeline runner
├── youtube_sports_analyzer.py    # CLI entry point
├── requirements.txt              # Python dependencies
│
├── modules/
│   ├── __init__.py
│   ├── downloader.py             # YouTube audio download via yt-dlp
│   ├── transcriber.py            # Whisper transcription with anti-hallucination params
│   ├── preprocessor.py           # Text normalization and sentence splitting
│   ├── event_detector.py         # 4-Tier hybrid event detection engine
│   ├── summarizer.py             # Results aggregation and narrative generation
│   └── reporter.py               # Report output (console + .txt file)
│
├── static/
│   ├── app.js                    # Frontend SSE client and UI logic
│   └── style.css                 # Styling
│
├── templates/
│   └── index.html                # Main web UI
│
└── reports/                      # Auto-generated analysis reports (gitignored)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip
- FFmpeg (bundled via `imageio-ffmpeg`, no manual install needed)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/YouTube-Sports-Analyzer.git
cd YouTube-Sports-Analyzer

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

> ⚠️ **Note:** PyTorch (`torch`) may require a specific install command depending on your system. Visit [pytorch.org](https://pytorch.org/get-started/locally/) if you encounter issues.

### Running the App

```bash
python app.py
```

Then open your browser at: **http://localhost:5000**

### Running via CLI

```bash
python youtube_sports_analyzer.py
```

---

## 🖥️ Usage

1. Open the web app at `http://localhost:5000`
2. Paste a YouTube sports video URL into the input field
3. Select a Whisper model size:
   - `tiny` — Fastest, least accurate (good for demos)
   - `base` — Balanced (recommended)
   - `small` / `medium` — Higher accuracy, slower
4. Click **Analyze**
5. Watch real-time pipeline progress updates
6. View the full event timeline, frequency table, key moments, and evaluation metrics
7. A `.txt` report is saved automatically to the `reports/` folder

---

## 📊 Sample Report Output

```
============================================================
YOUTUBE SPORTS COMMENTARY ANALYSIS REPORT
Generated: 2026-03-27 13:00:00
============================================================

[1] MATCH OVERVIEW
Total Sentences Analyzed: 312
Total Sports Events Detected: 47

[2] EVENT FREQUENCY
Event Type           | Frequency
GOAL                 | 6
GOALKEEPER_SAVE      | 11
FOUL                 | 9
CORNER_KICK          | 8
...

[5] EVALUATION METRICS (Auto-Computed from This Video)
Tier 1  (Keyword Phrases)             :  28 events  (59.6% of detections)
Tier 1.5 (Single-Word Regex)          :  10 events  (21.3% of detections)
Tier 1.75 (Numeric Scorelines)        :   3 events  ( 6.4% of detections)
Tier 2  (TF-IDF Label Spreading)      :   6 events  (12.8% of detections)

Average ML Confidence                 : 78.3%
Sentences Filtered as NEUTRAL         : 265 sentences auto-rejected
Active suppression rate               : 84.9% of input sentences
```

---

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Whisper model | `tiny` | Model size (tiny/base/small/medium) |
| ML confidence threshold | `0.65` | Minimum Label Spreading confidence to accept |
| Deduplication window | `10s` | Seconds between same-type event detections |
| Flask port | `5000` | Web server port |

---

## 📦 Dependencies

```
openai-whisper
yt-dlp
imageio-ffmpeg
soundfile
numpy
scikit-learn
flask
python-pptx
torch
librosa
```

---

## 🙌 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) — Speech recognition model
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube audio downloader
- [scikit-learn](https://scikit-learn.org/) — TF-IDF vectorization and Label Spreading

---

## 📄 License

This project is for academic and educational purposes.

---

## 👨‍💻 Developer Info

- **Developer:** santhosh kollu
- **Contact:** [kollusanthosh01@gmail.com](mailto:kollusanthosh01@gmail.com)
