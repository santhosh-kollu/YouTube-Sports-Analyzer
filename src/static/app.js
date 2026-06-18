document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const youtubeInput = document.getElementById('youtube-url');
    const modelSelect = document.getElementById('whisper-model');
    const pipelineSection = document.getElementById('pipeline-status');
    const resultsView = document.getElementById('results-view');
    const errorBox = document.getElementById('error-box');
    const errorMsg = document.getElementById('error-msg');

    const audioFileInput = document.getElementById('audio-file');
    const fileNameDisplay = document.getElementById('file-name-display');

    let eventSource = null;

    if (audioFileInput) {
        audioFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                fileNameDisplay.innerText = e.target.files[0].name;
                youtubeInput.value = ''; // Clear URL if file is selected
            } else {
                fileNameDisplay.innerText = '';
            }
        });
    }

    analyzeBtn.addEventListener('click', async () => {
        const url = youtubeInput.value.trim();
        const file = audioFileInput && audioFileInput.files.length > 0 ? audioFileInput.files[0] : null;

        if (!url && !file) {
            showError("Please enter a YouTube URL or upload an audio/video file.");
            return;
        }

        // Reset UI
        resultsView.classList.add('hidden');
        errorBox.classList.add('hidden');
        pipelineSection.classList.remove('hidden');
        resetTimelineIcons();
        
        // Disable button
        analyzeBtn.disabled = true;
        analyzeBtn.querySelector('.btn-text').innerText = file ? "Uploading..." : "Processing...";
        analyzeBtn.querySelector('.btn-loader').classList.remove('hidden');
        analyzeBtn.querySelector('.btn-icon').classList.add('hidden');

        try {
            const formData = new FormData();
            formData.append('model', modelSelect.value);
            
            let endpoint = '/analyze';
            
            if (file) {
                formData.append('file', file);
                endpoint = '/api/upload';
            } else {
                formData.append('url', url);
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || "Server failed to start analysis.");
            }

            // Start Listening for SSE
            startStreaming(data.session_id);

        } catch (err) {
            showError(err.message);
            resetButton();
        }
    });

    function startStreaming(sessionId) {
        if (eventSource) eventSource.close();
        
        eventSource = new EventSource(`/stream?session_id=${sessionId}`);
        
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log("SSE Update:", data);

            if (data.step === 'error') {
                showError(data.msg);
                updateStepUI(data.step, 'error', data.msg);
                eventSource.close();
                resetButton();
            } else {
                updateStepUI(data.step, data.status, data.msg);
                
                if (data.step === 'summarize' && data.status === 'done') {
                    renderResults(data.result, data.raw_text, data.detected_events || []);
                    eventSource.close();
                    resetButton();
                }
            }
        };

        eventSource.onerror = (err) => {
            console.error("SSE Error:", err);
            eventSource.close();
            resetButton();
        };
    }

    function updateStepUI(step, status, msg) {
        const stepEl = document.querySelector(`.pipeline-step[data-step="${step}"]`);
        if (!stepEl) return;

        stepEl.classList.remove('active', 'done', 'error');
        if (status === 'active') stepEl.classList.add('active');
        if (status === 'done') stepEl.classList.add('done');
        if (status === 'error') stepEl.classList.add('error');

        stepEl.querySelector('.step-status').innerText = msg;
    }

    function renderResults(result, rawText, detectedEvents) {
        // Show view
        resultsView.classList.remove('hidden');
        
        // Stats Boxes
        document.getElementById('stat-total-events').innerText = result.total_events;
        
        const goalCount = result.frequency_table.find(x => x[0] === 'GOAL')?.[1] || 0;
        document.getElementById('stat-goals').innerText = goalCount;
        
        const saveCount = result.frequency_table.find(x => x[0] === 'GOALKEEPER_SAVE')?.[1] || 0;
        document.getElementById('stat-saves').innerText = saveCount;

        const cardCount = (result.frequency_table.find(x => x[0] === 'YELLOW_CARD')?.[1] || 0) + 
                         (result.frequency_table.find(x => x[0] === 'RED_CARD')?.[1] || 0);
        document.getElementById('stat-cards').innerText = cardCount;

        // Frequency Bars
        const freqList = document.getElementById('frequency-list');
        freqList.innerHTML = '';
        const maxVal = Math.max(...result.frequency_table.map(x => x[1]), 1);
        
        result.frequency_table.forEach(([type, count]) => {
            const pct = (count / maxVal) * 100;
            const item = document.createElement('div');
            item.className = 'freq-item';
            item.innerHTML = `
                <div class="freq-label-row">
                    <span>${type.replace('_', ' ')}</span>
                    <span>${count}</span>
                </div>
                <div class="freq-bar-bg">
                    <div class="freq-bar-fill" style="width: 0%"></div>
                </div>
            `;
            freqList.appendChild(item);
            setTimeout(() => {
                item.querySelector('.freq-bar-fill').style.width = pct + '%';
            }, 100);
        });

        // Narrative
        document.getElementById('narrative-text').innerText = result.narrative;

        // Key Moments
        const momentsList = document.getElementById('key-moments-list');
        momentsList.innerHTML = '';
        result.key_moments.forEach(m => {
            const div = document.createElement('div');
            div.className = 'moment-item';
            div.style.marginBottom = '15px';
            div.innerHTML = `
                <div style="font-weight:700; color:var(--accent-blue); display:flex; gap:8px;">
                    <i class="fas fa-certificate"></i> ${m.type} (x${m.count})
                </div>
                <div style="font-size:0.85rem; color:var(--text-secondary); font-style:italic;">
                    "${m.sample}"
                </div>
            `;
            momentsList.appendChild(div);
        });

        // Timeline
        const timelineList = document.getElementById('timeline-list');
        timelineList.innerHTML = '';
        result.timeline.forEach(item => {
            const div = document.createElement('div');
            div.className = 'timeline-item';
            div.innerHTML = `
                <div class="timeline-time">${item.time}</div>
                <div class="timeline-content">
                    <div class="timeline-type">${item.type}</div>
                    <div class="timeline-text">${item.commentary}</div>
                </div>
            `;
            timelineList.appendChild(div);
        });

        // Raw Transcript
        document.getElementById('raw-transcript-view').innerText = rawText;

        // ── Evaluation Metrics (computed from live detected_events) ────────────
        if (detectedEvents && detectedEvents.length > 0) {
            const total = detectedEvents.length;
            const tier1 = detectedEvents.filter(e => (e.confidence || '').includes('tier 1'));
            const tier2 = detectedEvents.filter(e => (e.confidence || '').includes('tier 2'));
            const totalSentences = result.total_sentences || 0;

            // Parse confidence % from "ml (78%) tier 2"
            const tier2Confs = tier2.map(e => {
                const m = e.confidence.match(/\((\d+)%\)/);
                return m ? parseFloat(m[1]) : null;
            }).filter(v => v !== null);

            const avgConf = tier2Confs.length
                ? (tier2Confs.reduce((a, b) => a + b, 0) / tier2Confs.length).toFixed(1)
                : null;

            const suppressed = totalSentences - total;

            document.getElementById('eval-tier1').innerText = tier1.length;
            document.getElementById('eval-tier1-pct').innerText =
                `${((tier1.length / total) * 100).toFixed(1)}% of detections`;

            document.getElementById('eval-tier2').innerText = tier2.length;
            document.getElementById('eval-tier2-pct').innerText =
                `${((tier2.length / total) * 100).toFixed(1)}% of detections`;

            document.getElementById('eval-avg-conf').innerText =
                avgConf ? avgConf + '%' : 'N/A';

            document.getElementById('eval-suppressed').innerText = suppressed;
        }

        // Scroll to results
        resultsView.scrollIntoView({ behavior: 'smooth' });
    }

    function resetButton() {
        analyzeBtn.disabled = false;
        analyzeBtn.querySelector('.btn-text').innerText = "Analyze";
        analyzeBtn.querySelector('.btn-loader').classList.add('hidden');
        analyzeBtn.querySelector('.btn-icon').classList.remove('hidden');
    }

    function resetTimelineIcons() {
        document.querySelectorAll('.pipeline-step').forEach(step => {
            step.classList.remove('active', 'done', 'error');
            step.querySelector('.step-status').innerText = "Pending";
        });
    }

    function showError(msg) {
        errorMsg.innerText = msg;
        errorBox.classList.remove('hidden');
        errorBox.scrollIntoView({ behavior: 'smooth' });
    }
});
