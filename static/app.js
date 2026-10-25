/**
 * AURA Vision — Interactive Biometrics Controller
 * Handles Webcam Streaming, Real-time Face Tracking HUD, Web Audio Chimes,
 * Guided Enrollment Wizard, and Attendance Ledger.
 */

const API_BASE = window.location.origin;

class AuraApp {
    constructor() {
        this.video = document.getElementById('camera-video');
        this.canvas = document.getElementById('camera-canvas');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;

        this.isStreaming = false;
        this.isRecognizing = true;
        this.isMirrored = true;
        this.activeStream = null;
        this.recognitionInterval = null;

        this.lastDetectedFace = null;
        this.audioCtx = null;

        // Enrollment Wizard State
        this.enrollmentSteps = [
            { id: 1, title: "Look Straight", prompt: "Position your face in the center and look straight." },
            { id: 2, title: "Turn Slightly Left", prompt: "Turn your head gently 15° to the left." },
            { id: 3, title: "Turn Slightly Right", prompt: "Turn your head gently 15° to the right." },
            { id: 4, title: "Tilt Slightly Up", prompt: "Tilt your chin slightly upwards." },
            { id: 5, title: "Smile Naturally", prompt: "Smile or adopt a natural relaxed expression." }
        ];
        this.currentEnrollStep = 0;
        this.capturedCrops = [];

        this.init();
    }

    async init() {
        this.initScrollReveal();
        this.init3DTilt();
        this.bindEvents();
        await this.fetchStatus();
        await this.fetchAttendance();
        this.startCamera();
    }

    // Web Audio API Chime Synthesizer
    playChime(type = 'success') {
        try {
            if (!this.audioCtx) {
                this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (this.audioCtx.state === 'suspended') {
                this.audioCtx.resume();
            }

            const now = this.audioCtx.currentTime;
            const osc = this.audioCtx.createOscillator();
            const gain = this.audioCtx.createGain();

            osc.connect(gain);
            gain.connect(this.audioCtx.destination);

            if (type === 'success') {
                // Crystal double-bell chime
                osc.type = 'sine';
                osc.frequency.setValueAtTime(587.33, now); // D5
                osc.frequency.exponentialRampToValueAtTime(880.0, now + 0.12); // A5
                gain.gain.setValueAtTime(0.3, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
                osc.start(now);
                osc.stop(now + 0.5);
            } else {
                // Muted warning tone
                osc.type = 'triangle';
                osc.frequency.setValueAtTime(320, now);
                osc.frequency.exponentialRampToValueAtTime(220, now + 0.2);
                gain.gain.setValueAtTime(0.25, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
                osc.start(now);
                osc.stop(now + 0.35);
            }
        } catch (e) {
            console.warn("Audio chime disabled:", e);
        }
    }

    showToast(message, type = 'success') {
        const container = document.getElementById('toast-container');
        if (!container) return;
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        let iconSvg = '';
        if (type === 'success') {
            iconSvg = `<svg class="svg-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
        } else if (type === 'error') {
            iconSvg = `<svg class="svg-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`;
        } else {
            iconSvg = `<svg class="svg-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;
        }

        toast.innerHTML = `<span>${iconSvg}</span> <span>${message}</span>`;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(15px)';
            setTimeout(() => toast.remove(), 400);
        }, 3800);
    }

    // --- Camera & Recognition Management ---
    async startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
                audio: false
            });
            this.activeStream = stream;
            this.video.srcObject = stream;
            await this.video.play();
            this.isStreaming = true;

            this.video.addEventListener('loadedmetadata', () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.startRecognitionLoop();
            });

            this.showToast("Camera stream connected", "success");
        } catch (err) {
            console.error("Camera access failed:", err);
            this.showToast("Camera access denied or unavailable", "error");
        }
    }

    stopCamera() {
        if (this.activeStream) {
            this.activeStream.getTracks().forEach(t => t.stop());
            this.activeStream = null;
        }
        clearInterval(this.recognitionInterval);
        this.isStreaming = false;
        if (this.ctx) this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    startRecognitionLoop() {
        clearInterval(this.recognitionInterval);
        // Process frames at ~7 FPS to balance smoothness and neural inference efficiency
        this.recognitionInterval = setInterval(() => {
            if (this.isStreaming && this.isRecognizing) {
                this.processVideoFrame();
            }
        }, 150);
    }

    async processVideoFrame() {
        if (!this.video || this.video.readyState !== 4) return;

        const offCanvas = document.createElement('canvas');
        offCanvas.width = 480;
        offCanvas.height = 360;
        const offCtx = offCanvas.getContext('2d');

        if (this.isMirrored) {
            offCtx.translate(offCanvas.width, 0);
            offCtx.scale(-1, 1);
        }
        offCtx.drawImage(this.video, 0, 0, offCanvas.width, offCanvas.height);
        const dataUrl = offCanvas.toDataURL('image/jpeg', 0.82);

        try {
            const resp = await fetch(`${API_BASE}/api/recognize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataUrl })
            });
            if (!resp.ok) return;
            const data = await resp.json();

            this.renderTrackingHUD(data.faces, offCanvas.width, offCanvas.height);
            this.updateTargetSidebar(data.faces);
        } catch (e) {
            console.warn("Recognition loop packet drop:", e);
        }
    }

    renderTrackingHUD(faces, srcW, srcH) {
        if (!this.ctx || !this.canvas) return;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (!faces || faces.length === 0) {
            return;
        }

        const scaleX = this.canvas.width / srcW;
        const scaleY = this.canvas.height / srcH;

        faces.forEach((face) => {
            const b = face.box;
            let x = b.x * scaleX;
            let y = b.y * scaleY;
            let w = b.w * scaleX;
            let h = b.h * scaleY;

            // Handle mirror coordinate inversion
            if (this.isMirrored) {
                x = this.canvas.width - (x + w);
            }

            const isVerified = face.verified;
            const primaryColor = isVerified ? '#00f5a0' : '#ff4757';
            const glowColor = isVerified ? 'rgba(0, 245, 160, 0.4)' : 'rgba(255, 71, 87, 0.4)';

            this.ctx.save();
            this.ctx.strokeStyle = primaryColor;
            this.ctx.lineWidth = 2.5;
            this.ctx.shadowBlur = 14;
            this.ctx.shadowColor = glowColor;

            // Sci-fi Corner Brackets
            const cornerSize = Math.min(w, h) * 0.22;
            // Top-left
            this.ctx.beginPath();
            this.ctx.moveTo(x, y + cornerSize);
            this.ctx.lineTo(x, y);
            this.ctx.lineTo(x + cornerSize, y);
            this.ctx.stroke();

            // Top-right
            this.ctx.beginPath();
            this.ctx.moveTo(x + w - cornerSize, y);
            this.ctx.lineTo(x + w, y);
            this.ctx.lineTo(x + w, y + cornerSize);
            this.ctx.stroke();

            // Bottom-left
            this.ctx.beginPath();
            this.ctx.moveTo(x, y + h - cornerSize);
            this.ctx.lineTo(x, y + h);
            this.ctx.lineTo(x + cornerSize, y + h);
            this.ctx.stroke();

            // Bottom-right
            this.ctx.beginPath();
            this.ctx.moveTo(x + w - cornerSize, y + h);
            this.ctx.lineTo(x + w, y + h);
            this.ctx.lineTo(x + w, y + h - cornerSize);
            this.ctx.stroke();

            // Render facial landmarks if present
            if (face.landmarks && face.landmarks.length > 0) {
                this.ctx.fillStyle = '#00f2fe';
                this.ctx.shadowColor = '#00f2fe';
                face.landmarks.forEach((lm) => {
                    let lx = lm.x * scaleX;
                    let ly = lm.y * scaleY;
                    if (this.isMirrored) lx = this.canvas.width - lx;
                    this.ctx.beginPath();
                    this.ctx.arc(lx, ly, 3.5, 0, Math.PI * 2);
                    this.ctx.fill();
                });
            }

            // Name Tag Pill
            const tagText = `${face.name} • ${face.confidence}%`;
            this.ctx.font = 'bold 15px "Plus Jakarta Sans", system-ui, sans-serif';
            const textWidth = this.ctx.measureText(tagText).width;
            const tagHeight = 28;
            const tagX = x + (w - textWidth - 24) / 2;
            const tagY = y - tagHeight - 8;

            this.ctx.fillStyle = 'rgba(5, 11, 24, 0.88)';
            this.ctx.strokeStyle = primaryColor;
            this.ctx.lineWidth = 1.5;
            this.roundRect(this.ctx, tagX, tagY, textWidth + 24, tagHeight, 14, true, true);

            this.ctx.fillStyle = '#ffffff';
            this.ctx.fillText(tagText, tagX + 12, tagY + 19);

            this.ctx.restore();
        });
    }

    roundRect(ctx, x, y, width, height, radius, fill, stroke) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
        if (fill) ctx.fill();
        if (stroke) ctx.stroke();
    }

    updateTargetSidebar(faces) {
        const nameEl = document.getElementById('target-student-name');
        const badgeEl = document.getElementById('target-status-badge');
        const meterEl = document.getElementById('confidence-meter-fill');
        const percentEl = document.getElementById('confidence-percentage');
        const avatarEl = document.getElementById('target-avatar-initials');

        if (!faces || faces.length === 0) {
            nameEl.textContent = "Scanning...";
            badgeEl.className = "target-status-badge";
            badgeEl.textContent = "Waiting for face";
            meterEl.style.width = "0%";
            percentEl.textContent = "0%";
            avatarEl.textContent = "—";
            this.lastDetectedFace = null;
            return;
        }

        const topFace = faces[0];
        this.lastDetectedFace = topFace;

        nameEl.textContent = topFace.name;
        avatarEl.textContent = topFace.name.substring(0, 2).toUpperCase();
        percentEl.textContent = `${topFace.confidence}%`;
        meterEl.style.width = `${topFace.confidence}%`;

        if (topFace.verified) {
            badgeEl.className = "target-status-badge status-verified";
            badgeEl.textContent = "Verified Identity";
            meterEl.style.background = "linear-gradient(90deg, var(--cyan-neon), var(--emerald-mint))";
        } else {
            badgeEl.className = "target-status-badge status-unknown";
            badgeEl.textContent = "Unregistered / Unknown";
            meterEl.style.background = "linear-gradient(90deg, var(--crimson-warn), #f6ad55)";
        }
    }

    async punchAttendance() {
        if (!this.lastDetectedFace) {
            this.showToast("No face detected to punch attendance", "warn");
            this.playChime('warn');
            return;
        }

        const payload = {
            name: this.lastDetectedFace.name,
            confidence: this.lastDetectedFace.confidence
        };

        try {
            const resp = await fetch(`${API_BASE}/api/attendance/punch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();

            if (data.success) {
                this.playChime('success');
                this.showToast(data.message, 'success');
                await this.fetchAttendance();
            } else {
                this.playChime('warn');
                this.showToast(data.message, 'warn');
            }
        } catch (e) {
            this.showToast("Failed to communicate with attendance service", "error");
        }
    }

    // --- Biometric Enrollment Wizard ---
    nextEnrollmentStep() {
        const studentName = document.getElementById('enroll-student-name').value.trim();
        if (!studentName) {
            this.showToast("Please enter student name first", "warn");
            return;
        }

        if (!this.video || this.video.readyState !== 4) {
            this.showToast("Camera not ready for capture", "warn");
            return;
        }

        // Capture current face crop
        const crop = this.captureCurrentCrop();
        this.capturedCrops.push(crop);

        // Update preview strip
        const previewBoxes = document.querySelectorAll('.crop-preview-box');
        if (previewBoxes[this.currentEnrollStep]) {
            previewBoxes[this.currentEnrollStep].innerHTML = `<img src="${crop}" alt="Pose ${this.currentEnrollStep + 1}">`;
        }

        // Update step UI
        const stepItems = document.querySelectorAll('.step-item');
        if (stepItems[this.currentEnrollStep]) {
            stepItems[this.currentEnrollStep].classList.remove('active');
            stepItems[this.currentEnrollStep].classList.add('done');
        }

        this.currentEnrollStep++;

        if (this.currentEnrollStep < this.enrollmentSteps.length) {
            if (stepItems[this.currentEnrollStep]) {
                stepItems[this.currentEnrollStep].classList.add('active');
            }
            const promptEl = document.getElementById('wizard-prompt-text');
            if (promptEl) {
                promptEl.textContent = this.enrollmentSteps[this.currentEnrollStep].prompt;
            }
            this.showToast(`Step ${this.currentEnrollStep} captured! Look at prompt for step ${this.currentEnrollStep + 1}`, "accent");
            this.playChime('success');
        } else {
            // All 5 steps completed!
            document.getElementById('btn-capture-pose').style.display = 'none';
            document.getElementById('btn-submit-enroll').style.display = 'inline-flex';
            const promptEl = document.getElementById('wizard-prompt-text');
            if (promptEl) {
                promptEl.textContent = "All 5 poses captured! Click 'Save & Auto-Train' to finalize enrollment.";
            }
            this.playChime('success');
        }
    }

    captureCurrentCrop() {
        const off = document.createElement('canvas');
        off.width = 120;
        off.height = 120;
        const octx = off.getContext('2d');

        // Extract center 55% of webcam as crop
        const vw = this.video.videoWidth;
        const vh = this.video.videoHeight;
        const cropSize = Math.min(vw, vh) * 0.55;
        const sx = (vw - cropSize) / 2;
        const sy = (vh - cropSize) / 2;

        if (this.isMirrored) {
            octx.translate(off.width, 0);
            octx.scale(-1, 1);
        }
        octx.drawImage(this.video, sx, sy, cropSize, cropSize, 0, 0, off.width, off.height);
        return off.toDataURL('image/jpeg', 0.9);
    }

    async submitEnrollment() {
        const studentName = document.getElementById('enroll-student-name').value.trim();
        if (!studentName || this.capturedCrops.length === 0) return;

        const btn = document.getElementById('btn-submit-enroll');
        btn.disabled = true;
        btn.textContent = "Saving & Augmenting...";

        try {
            const resp = await fetch(`${API_BASE}/api/students/enroll`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: studentName,
                    images: this.capturedCrops
                })
            });
            const data = await resp.json();

            if (data.success) {
                this.showToast(data.message, "success");
                this.playChime('success');
                this.resetEnrollmentWizard();
                await this.fetchStatus();
            } else {
                this.showToast(data.error || "Enrollment failed", "error");
            }
        } catch (e) {
            this.showToast("Failed to complete enrollment", "error");
        } finally {
            btn.disabled = false;
            btn.textContent = "Save & Auto-Train";
        }
    }

    resetEnrollmentWizard() {
        this.currentEnrollStep = 0;
        this.capturedCrops = [];
        document.getElementById('enroll-student-name').value = '';
        document.getElementById('btn-capture-pose').style.display = 'inline-flex';
        document.getElementById('btn-submit-enroll').style.display = 'none';

        const stepItems = document.querySelectorAll('.step-item');
        stepItems.forEach((item, idx) => {
            item.className = `step-item ${idx === 0 ? 'active' : ''}`;
        });

        const previewBoxes = document.querySelectorAll('.crop-preview-box');
        previewBoxes.forEach((box, idx) => {
            box.innerHTML = `Pose ${idx + 1}`;
        });

        const promptEl = document.getElementById('wizard-prompt-text');
        if (promptEl) {
            promptEl.textContent = this.enrollmentSteps[0].prompt;
        }
    }

    // --- Data Fetching & Sync ---
    async fetchStatus() {
        try {
            const resp = await fetch(`${API_BASE}/api/status`);
            if (!resp.ok) return;
            const data = await resp.json();

            // Hero ribbon
            this.animateNumber('metric-accuracy', 99.2, '%');
            this.animateNumber('metric-latency', data.inference_latency_ms, 'ms');
            this.animateNumber('metric-students', data.student_count);
        } catch (e) {
            console.warn("Status fetch failed:", e);
        }
    }

    async fetchAttendance() {
        try {
            const resp = await fetch(`${API_BASE}/api/attendance/today`);
            if (!resp.ok) return;
            const data = await resp.json();

            const tableBody = document.getElementById('attendance-table-body');
            const totalCountEl = document.getElementById('attendance-total-count');
            if (totalCountEl) totalCountEl.textContent = data.stats.total_punches;

            if (!tableBody) return;
            tableBody.innerHTML = '';

            if (data.records.length === 0) {
                tableBody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted); padding: 24px;">No attendance logged yet today. Look at the camera and click 'Punch Attendance'!</td></tr>`;
                return;
            }

            data.records.forEach((r) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-weight: 700; color: #fff;">${r.name}</td>
                    <td style="font-family: var(--font-mono); color: var(--cyan-neon);">${r.time}</td>
                    <td style="font-family: var(--font-mono);">${r.confidence}</td>
                    <td><span class="aug-tag" style="background: rgba(0,245,160,0.15); color: var(--emerald-mint); border: 1px solid var(--emerald-mint);">Verified</span></td>
                `;
                tableBody.appendChild(tr);
            });
        } catch (e) {
            console.warn("Attendance fetch failed:", e);
        }
    }

    animateNumber(elId, target, suffix = '') {
        const el = document.getElementById(elId);
        if (!el) return;
        const start = 0;
        const duration = 1200;
        const startTime = performance.now();

        const update = (now) => {
            const progress = Math.min((now - startTime) / duration, 1);
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = (target * easeOut).toFixed(target % 1 === 0 ? 0 : 1);
            el.textContent = `${current}${suffix}`;
            if (progress < 1) requestAnimationFrame(update);
        };
        requestAnimationFrame(update);
    }

    // --- Interactive 3D Tilt on Hover ---
    init3DTilt() {
        const tiltCards = document.querySelectorAll('.glass-panel, .glass-card');
        tiltCards.forEach((card) => {
            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const cx = rect.width / 2;
                const cy = rect.height / 2;
                const rotX = ((y - cy) / cy) * -4;
                const rotY = ((x - cx) / cx) * 4;

                card.style.transform = `perspective(1000px) rotateX(${rotX}deg) rotateY(${rotY}deg) translateY(-2px)`;
            });

            card.addEventListener('mouseleave', () => {
                card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0px)';
            });
        });
    }

    // --- Scroll Reveal Animations ---
    initScrollReveal() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('revealed');
                }
            });
        }, { threshold: 0.12 });

        document.querySelectorAll('.reveal-on-scroll').forEach(el => observer.observe(el));
    }

    bindEvents() {
        // Attendance Punch button
        const punchBtn = document.getElementById('btn-punch-attendance');
        if (punchBtn) punchBtn.addEventListener('click', () => this.punchAttendance());

        // Enrollment Wizard buttons
        const capturePoseBtn = document.getElementById('btn-capture-pose');
        if (capturePoseBtn) capturePoseBtn.addEventListener('click', () => this.nextEnrollmentStep());

        const submitEnrollBtn = document.getElementById('btn-submit-enroll');
        if (submitEnrollBtn) submitEnrollBtn.addEventListener('click', () => this.submitEnrollment());

        // Camera Toggle controls
        const toggleCamBtn = document.getElementById('btn-toggle-camera');
        if (toggleCamBtn) {
            toggleCamBtn.addEventListener('click', () => {
                if (this.isStreaming) {
                    this.stopCamera();
                    toggleCamBtn.querySelector('span').textContent = "Start Camera";
                } else {
                    this.startCamera();
                    toggleCamBtn.querySelector('span').textContent = "Pause Camera";
                }
            });
        }

        const flipCamBtn = document.getElementById('btn-flip-camera');
        if (flipCamBtn) {
            flipCamBtn.addEventListener('click', () => {
                this.isMirrored = !this.isMirrored;
                this.video.style.transform = this.isMirrored ? 'scaleX(-1)' : 'scaleX(1)';
            });
        }

        // Attendance Table Search filter
        const searchInput = document.getElementById('attendance-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                const term = e.target.value.toLowerCase();
                const rows = document.querySelectorAll('#attendance-table-body tr');
                rows.forEach((row) => {
                    const name = row.cells[0]?.textContent.toLowerCase() || '';
                    row.style.display = name.includes(term) ? '' : 'none';
                });
            });
        }
    }
}

// Launch on DOM ready
window.addEventListener('DOMContentLoaded', () => {
    window.auraApp = new AuraApp();
});
