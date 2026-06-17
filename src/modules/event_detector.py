import logging
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.semi_supervised import LabelSpreading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Phrase Keyword Matching (High Confidence)
#
# A manually constructed dictionary mapping 11 event types to extensive lists
# of multi-word phrases. Phrases are used instead of single words to eliminate
# false positives (e.g., "back of the net" is unambiguous; "net" alone is not).
# Matching is performed on lowercase, punctuation-stripped, stopword-removed text.
# ─────────────────────────────────────────────────────────────────────────────
EVENT_KEYWORDS = {
    "GOAL": [
        "what a goal", "goal goal goal", "back of the net",
        "puts it away", "its a goal", "scores the goal",
        "tucks it in", "nets it", "into the net", "hes scored",
        "brilliant goal", "stunning goal", "header goal", "volley goal",
        "the goal is scored", "and its in", "sweeps home", "fires home",
        "two nil", "one nil", "three nil", "one one", "two one", "three one"
    ],
    "FOUL": [
        "that is a foul", "foul by", "clear foul", "taken down",
        "the player is fouled", "trip by", "he trips", "foul committed",
        "that was a foul", "dangerous foul", "reckless challenge",
        "cynical foul", "brings him down"
    ],
    "PENALTY": [
        "penalty kick", "points to the spot", "from the penalty spot",
        "penalty is awarded", "it is a penalty", "penalty decision",
        "handball penalty", "spot kick", "twelve yard", "stonewall penalty",
        "referee points to spot"
    ],
    "YELLOW_CARD": [
        "yellow card", "shown the yellow", "receives a caution",
        "booked for that", "into the book", "referee books",
        "cautioned by the referee", "picks up a booking"
    ],
    "RED_CARD": [
        "red card", "sent off", "shown the red",
        "direct red", "off the pitch now", "early shower",
        "dismissed from the field", "straight red"
    ],
    "CORNER_KICK": [
        "corner kick", "from the corner flag", "taking the corner",
        "whips in the corner", "corner is awarded", "into the box from the corner",
        "ball goes out for a corner"
    ],
    "OFFSIDE": [
        "flag is up for offside", "caught offside", "offside flag",
        "linesman raises the flag", "ruled offside", "offside decision",
        "in an offside position", "goal ruled out offside"
    ],
    "GOALKEEPER_SAVE": [
        "what a save", "brilliant save", "incredible save", "keeper saves",
        "goalkeeper saves", "tips it over", "punches it clear",
        "goalkeeper denies", "reflex save", "great stop by the keeper",
        "keeper is equal to it", "pulled off a save"
    ],
    "SUBSTITUTION": [
        "substitution is being made", "being substituted", "coming on for",
        "going off for", "changes on the board", "player is subbed",
        "the manager changes", "fresh legs"
    ],
    "FREE_KICK": [
        "free kick awarded", "awarded a free kick", "free kick from",
        "takes the free kick", "curls the free kick", "free kick outside the box",
        "direct free kick"
    ],
    "NEAR_MISS": [
        "just wide", "hits the post", "hits the crossbar", "so close",
        "narrowly misses", "inches wide", "off the bar",
        "goes just over", "just over the bar", "agonizingly close",
        "rattles the post", "off the woodwork", "clips the bar"
    ]
}

# ─────────────────────────────────────────────────────────────────────────────
# Tier 1.5 — Single-Word Regex Marker Matching (Medium-High Confidence)
#
# Uses Python re module with word-boundary anchors (\b) to match single
# decisive words that are NOT captured by multi-word phrases in Tier 1.
#
# The \b anchor is critical:
#   \bgoal\b   → matches "goal" but NOT "goalkeeper", "goalie", "goalpost"
#   \bsave\b   → matches "save" but NOT "saved", "goalkeeper save" (Tier 1)
#   \bfoul\b   → matches "foul" but NOT "fowl", "foulest"
#
# Only fires on sentences NOT already classified by Tier 1.
# ─────────────────────────────────────────────────────────────────────────────
SINGLE_WORD_MARKERS = {
    "GOAL": [
        r'\bgoal\b',
        r'\bscores\b',
        r'\bscored\b',
        r'\bnets\b',
        r'\bfinishes\b',
        r'\bfinished\b',
    ],
    "FOUL": [
        r'\bfoul\b',
        r'\btackled\b',
        r'\bfouled\b',
        r'\btripped\b',
        r'\bclattered\b',
    ],
    "PENALTY": [
        r'\bpenalty\b',
    ],
    "YELLOW_CARD": [
        r'\bbooked\b',
        r'\bcautioned\b',
    ],
    "RED_CARD": [
        r'\bdismissed\b',
    ],
    "CORNER_KICK": [
        r'\bcorner\b',
    ],
    "OFFSIDE": [
        r'\boffside\b',
    ],
    "GOALKEEPER_SAVE": [
        r'\bsave\b',
        r'\bdenied\b',
        r'\bblocked\b',
        r'\bparried\b',
        r'\bpalmed\b',
    ],
    "SUBSTITUTION": [
        r'\bsubstitution\b',
        r'\bsubstitute\b',
        r'\bsubbed\b',
    ],
    "FREE_KICK": [
        r'\bfreekick\b',
        r'\bfree-kick\b',
    ],
    "NEAR_MISS": [
        r'\bcrossbar\b',
        r'\bwoodwork\b',
        r'\bpost\b',
    ]
}

# ─────────────────────────────────────────────────────────────────────────────
# Tier 1.75 — Numeric Scoreline Pattern Matching (Medium Confidence)
#
# Sports commentary frequently announces scores numerically.
# A scoreline announcement is a highly reliable predictor of a GOAL event,
# even when no explicit goal-related vocabulary is present.
#
# Patterns cover:
#   Digit format:  "2-1",  "0-0",  "3 - 2"
#   Word format:   "two nil",  "one one",  "three-two"
#   Contextual:    "makes it 2",  "now 3-0",  "level at one"
#
# Only fires on sentences NOT already classified by Tiers 1 or 1.5.
# ─────────────────────────────────────────────────────────────────────────────
_NUM_WORDS = r'(one|two|three|four|five|six)'
_ZERO_WORDS = r'(nil|zero|nought)'
_ANY_NUM = rf'({_NUM_WORDS}|{_ZERO_WORDS}|\d+)'

SCORE_PATTERNS = [
    r'\b\d+\s*[-\u2013]\s*\d+\b',                              # "2-1",  "3 - 0"
    rf'\b{_NUM_WORDS}\s*[-\u2013]\s*{_ANY_NUM}\b',             # "two-nil", "two-one"
    rf'\b{_NUM_WORDS}\s+{_ANY_NUM}\b',                         # "two nil", "one one"
    rf'\b{_ZERO_WORDS}\s*[-\u2013]\s*{_ZERO_WORDS}\b',         # "nil-nil"
    r'\bmakes\s+it\s+\d+\b',                                   # "makes it 2"
    r'\bnow\s+\d+\s*[-\u2013]\s*\d+\b',                        # "now 2-1"
    r'\blevel\s+at\s+\d+\b',                                   # "level at 1"
    r'\b\d+\s+goals?\s+to\s+\d+\b',                            # "2 goals to 1"
    r'\bequalis[e|z]s?\b',                                      # "equalises"
    r'\bmakes\s+it\s+(one|two|three|four|five|nil|zero)\b',    # "makes it two"
]

# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Semi-Supervised Label Spreading (Lower Confidence)
#
# For sentences that passed through ALL three rule-based tiers unclassified,
# TF-IDF vectors are computed and Label Spreading propagates labels from
# hand-written seed examples to unclassified sentences based on vector
# similarity. A confidence threshold of 0.65 prevents uncertain predictions
# from becoming false positives.
# ─────────────────────────────────────────────────────────────────────────────
MIN_CONFIDENCE = 0.65

NEUTRAL_SEEDS = [
    "the ball is played across the midfield",
    "the team is pressing high up the pitch",
    "a long ball forward into the channel",
    "possession is won back by the home side",
    "both teams are set for the restart",
    "a good passage of play from the home side",
    "the players are getting into position",
    "crowd is watching intently in the stadium",
    "we are in the second half of the match",
    "the manager looks on from the touchline",
    "time is ticking down in this match",
    "the ball goes out for a throw in",
    "the winger cuts inside towards the box",
    "players are tracking back to defend",
    "the team is holding their shape at the back",
    "a header clearance from the defender",
    "good build up play from the midfield",
    "the striker holds the ball up",
    "the referee has a word with the players",
    "we are deep into stoppage time",
]

SEED_EXAMPLES = {
    "GOAL": [
        "what a goal by the striker into the back of the net",
        "he scores and makes it one nil",
        "brilliant finish and it is a goal",
        "puts it away with such composure"
    ],
    "FOUL": [
        "that is a clear foul by the defender on the winger",
        "the player is fouled right outside the area"
    ],
    "PENALTY": [
        "the referee points to the spot it is a penalty",
        "penalty kick is awarded for the handball"
    ],
    "YELLOW_CARD": [
        "the referee shows him the yellow card for that challenge",
        "he is booked for the foul"
    ],
    "RED_CARD": [
        "it is a straight red card for the defender",
        "he is sent off for that reckless tackle"
    ],
    "CORNER_KICK": [
        "taking the corner kick from the left flag",
        "the corner kick is whipped into the penalty area"
    ],
    "OFFSIDE": [
        "the linesman raises the flag for offside",
        "the goal is ruled out because he was in an offside position"
    ],
    "GOALKEEPER_SAVE": [
        "what a brilliant save by the goalkeeper to deny the striker",
        "the keeper tips it over the bar for a corner"
    ],
    "SUBSTITUTION": [
        "the manager is making a substitution and bringing on a fresh player",
        "the winger is coming off and a defender is going on"
    ],
    "FREE_KICK": [
        "taking the free kick from just outside the penalty area",
        "he curls the free kick beautifully towards the top corner"
    ],
    "NEAR_MISS": [
        "he hits the post from close range so close",
        "that just goes narrowly wide of the far post"
    ]
}


def detect_events(processed_data):
    """
    Detects sports events using a four-tier hybrid NLP approach.

    Tier 1   — Multi-word keyword phrase matching         (High confidence)
    Tier 1.5 — Single-word regex with word boundaries     (Medium-High confidence)
    Tier 1.75— Numeric scoreline pattern matching          (Medium confidence)
    Tier 2   — TF-IDF + Semi-Supervised Label Spreading   (Lower confidence, threshold-gated)
    """
    sentences         = processed_data["sentences"]
    cleaned_sentences = processed_data["cleaned_sentences"]
    timestamps        = processed_data["timestamps"]

    num_sentences = len(sentences)
    # -1 = unclassified. Stores integer index of event category once classified.
    tier1_labels  = np.full(num_sentences, -1)

    event_counts = {k: 0 for k in EVENT_KEYWORDS.keys()}
    event_counts["NEUTRAL"] = 0

    detected_events = []
    category_list   = list(EVENT_KEYWORDS.keys())

    # ── Tier 1: Multi-Word Keyword Phrase Matching ────────────────────────────
    logger.info("Starting Tier 1: Keyword Phrase Matching...")

    for idx, clean_sent in enumerate(cleaned_sentences):
        matched_event = None
        for event, keywords in EVENT_KEYWORDS.items():
            if any(kw in clean_sent for kw in keywords):
                matched_event = event
                break

        if matched_event:
            tier1_labels[idx] = category_list.index(matched_event)
            event_counts[matched_event] += 1
            detected_events.append({
                "index":      idx,
                "event":      matched_event,
                "timestamp":  timestamps[idx],
                "confidence": "high (tier 1)"
            })

    tier1_count = int(sum(tier1_labels != -1))
    logger.info(f"Tier 1 found {tier1_count} events.")

    # ── Tier 1.5: Single-Word Regex Marker Matching ───────────────────────────
    logger.info("Starting Tier 1.5: Single-Word Regex Matching...")

    tier15_count = 0
    for idx, clean_sent in enumerate(cleaned_sentences):
        if tier1_labels[idx] != -1:
            continue  # Already classified by Tier 1 — skip

        matched_event = None
        for event, patterns in SINGLE_WORD_MARKERS.items():
            if any(re.search(pattern, clean_sent) for pattern in patterns):
                matched_event = event
                break

        if matched_event:
            tier1_labels[idx] = category_list.index(matched_event)
            event_counts[matched_event] += 1
            detected_events.append({
                "index":      idx,
                "event":      matched_event,
                "timestamp":  timestamps[idx],
                "confidence": "medium (tier 1.5)"
            })
            tier15_count += 1

    logger.info(f"Tier 1.5 added {tier15_count} events.")

    # ── Tier 1.75: Numeric Scoreline Pattern Matching ─────────────────────────
    logger.info("Starting Tier 1.75: Numeric Scoreline Matching...")

    tier175_count = 0
    for idx, clean_sent in enumerate(cleaned_sentences):
        if tier1_labels[idx] != -1:
            continue  # Already classified — skip

        # Use the original sentence too for numeric matching (numbers may be stripped by stopword removal)
        original_sent = sentences[idx].lower()

        if any(re.search(pat, clean_sent) or re.search(pat, original_sent)
               for pat in SCORE_PATTERNS):
            tier1_labels[idx] = category_list.index("GOAL")
            event_counts["GOAL"] += 1
            detected_events.append({
                "index":      idx,
                "event":      "GOAL",
                "timestamp":  timestamps[idx],
                "confidence": "score (tier 1.75)"
            })
            tier175_count += 1

    logger.info(f"Tier 1.75 added {tier175_count} events.")

    # ── Tier 2: Semi-Supervised Label Spreading ───────────────────────────────
    logger.info("Starting Tier 2: Label Spreading with confidence threshold...")

    seed_texts     = []
    seed_label_ids = []
    neutral_idx    = len(category_list)   # NEUTRAL class index = 11

    for category, examples in SEED_EXAMPLES.items():
        cat_idx = category_list.index(category)
        for ex in examples:
            seed_texts.append(ex.lower())
            seed_label_ids.append(cat_idx)

    for neutral_ex in NEUTRAL_SEEDS:
        seed_texts.append(neutral_ex.lower())
        seed_label_ids.append(neutral_idx)

    all_texts      = cleaned_sentences + seed_texts
    initial_labels = np.concatenate([tier1_labels, seed_label_ids])

    if len(all_texts) > 0:
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000, sublinear_tf=True)
            X = vectorizer.fit_transform(all_texts)

            lp_model = LabelSpreading(kernel='knn', n_neighbors=5, alpha=0.2, max_iter=100)
            lp_model.fit(X.toarray(), initial_labels)

            pred_labels = lp_model.transduction_[:num_sentences]
            pred_probs  = lp_model.label_distributions_[:num_sentences]

            tier2_count = 0
            for i in range(num_sentences):
                if tier1_labels[i] != -1:
                    continue  # Already classified by Tiers 1, 1.5, or 1.75

                lbl_idx    = int(pred_labels[i])
                confidence = float(pred_probs[i][lbl_idx])

                if lbl_idx < len(category_list) and confidence >= MIN_CONFIDENCE:
                    event = category_list[lbl_idx]
                    event_counts[event] += 1
                    detected_events.append({
                        "index":      i,
                        "event":      event,
                        "timestamp":  timestamps[i],
                        "confidence": f"ml ({confidence:.0%}) tier 2"
                    })
                    tier2_count += 1
                else:
                    event_counts["NEUTRAL"] += 1

            logger.info(f"Tier 2 added {tier2_count} events (confidence >= {MIN_CONFIDENCE:.0%}).")

        except Exception as e:
            logger.error(f"Error during Tier 2 Label Spreading: {e}")

    logger.info(f"Event detection complete (before dedup). Total events: {len(detected_events)}")

    # ── Deduplication: Remove same-type events within 10-second window ─────────
    # When a goal (or any event) is scored, a commentator produces multiple
    # excited sentences within a few seconds — all describing the SAME moment.
    # Without this step, one goal gets counted 3–5 times and inflates the report.
    detected_events = _deduplicate_events(detected_events, window_seconds=10)

    # Recompute event_counts from deduplicated list for accurate frequency table
    event_counts = {k: 0 for k in EVENT_KEYWORDS.keys()}
    event_counts["NEUTRAL"] = 0
    for e in detected_events:
        event_counts[e["event"]] += 1

    logger.info(f"After deduplication. Final events: {len(detected_events)}")

    return {
        "detected_events": detected_events,
        "event_counts":    event_counts,
        "total_sentences": num_sentences
    }


def _deduplicate_events(detected_events, window_seconds=10):
    """
    Removes duplicate detections of the same event type within a sliding
    time window of `window_seconds` seconds.

    Rationale: A single real-world event (e.g., a goal) triggers multiple
    excited commentary sentences in quick succession. Each sentence is
    correctly detected as the same event type, but they all refer to ONE
    moment. Deduplication keeps only the FIRST detection per event type
    per time window, preventing inflated counts in the final report.

    Algorithm:
      1. Sort all events by their start timestamp (earliest first).
      2. For each event, check when the same event type was last accepted.
      3. If the gap is less than window_seconds → duplicate → discard.
      4. If the gap is >= window_seconds (or first occurrence) → keep.
    """
    if not detected_events:
        return detected_events

    # Step 1: Sort by start time so we always process earliest event first
    sorted_events = sorted(
        detected_events,
        key=lambda e: e["timestamp"]["start"]
    )

    # Step 2: Track last accepted timestamp per event type
    last_accepted_time = {}   # { "GOAL": 32.1, "FOUL": 61.4, ... }
    deduplicated = []

    for event in sorted_events:
        event_type   = event["event"]
        current_time = event["timestamp"]["start"]

        if event_type in last_accepted_time:
            gap = current_time - last_accepted_time[event_type]
            if gap < window_seconds:
                # Same event type appeared within the window → duplicate, skip
                logger.debug(
                    f"Dedup: Skipping {event_type} at {current_time:.1f}s "
                    f"(gap={gap:.1f}s < {window_seconds}s window)"
                )
                continue

        # Accept this event — it is either the first occurrence of this type
        # or it occurred far enough after the previous one to be a new event
        last_accepted_time[event_type] = current_time
        deduplicated.append(event)

    removed = len(sorted_events) - len(deduplicated)
    logger.info(f"Deduplication removed {removed} duplicate detections "
                f"(window={window_seconds}s). Kept {len(deduplicated)} events.")

    return deduplicated
