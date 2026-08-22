/**
 * IsharaConnect - Real-time Two-Way Split Dashboard Controller v2.0
 *
 * Architecture:
 *   LEFT  panel: Camera → WebRTC → CSLR ONNX → subtitle ticker + TTS vocalization
 *   RIGHT panel: Mic/Text → STT → WebSocket room → 3D Avatar sign playback
 *
 * WebSocket message protocol:
 *   Inbound  { type: "cslr_prediction", gloss, text, confidence, latency_ms }
 *   Inbound  { type: "speaker_text",    text }
 *   Outbound { type: "speaker_text",    text }
 *   Outbound { type: "ping" }
 */

class DashboardController {
  constructor() {
    this.config = window.BDSL_CONFIG || { roomId: "room-general", clientType: "signer" };
    this.transcriptLogs = [];
    this.currentText = "";
    this.isRecording = false;
    this._wsReconnectDelay = 1500;
    this._wsReconnectTimer = null;
    this._pingInterval = null;
    this._ttsQueue = [];
    this._ttsBusy = false;

    this.initDOMElements();
    this.initAvatar();
    this.initCamera();
    this.initWebSocket();
    this.initSpeechRecognition();
    this.initEventListeners();
    this._startPingLoop();
  }

  // ─── DOM ────────────────────────────────────────────────────────────────────

  initDOMElements() {
    this.localVideo      = document.getElementById("local-video");
    this.landmarkCanvas  = document.getElementById("landmark-canvas");
    this.signerSubtitle  = document.getElementById("signer-subtitle");
    this.confTag         = document.getElementById("confidence-tag");
    this.latencyTag      = document.getElementById("latency-tag");
    this.speakerInput    = document.getElementById("speaker-text-input");
    this.btnSend         = document.getElementById("btn-send-synthesize");
    this.btnMic          = document.getElementById("btn-voice-stt");
    this.btnTTS          = document.getElementById("btn-tts-speak");
    this.btnExport       = document.getElementById("btn-export-transcript");
    this.btnToggleCam    = document.getElementById("btn-toggle-cam");
    this.roleSelect      = document.getElementById("role-select");
    this.connBadge       = document.getElementById("conn-badge");
    this.waveformCanvas  = document.getElementById("audio-waveform");
    this.cslrFeedBadge   = document.getElementById("cslr-feed-badge");
    this.transcriptList  = document.getElementById("transcript-list");
  }

  // ─── Avatar ─────────────────────────────────────────────────────────────────

  initAvatar() {
    if (typeof HumanoidBdSLAvatar !== "undefined") {
      this.avatar = new HumanoidBdSLAvatar("avatar-container");
    } else if (typeof Ishara3DFACSAvatar !== "undefined") {
      this.avatar = new Ishara3DFACSAvatar("avatar-container");
    }
  }

  // ─── Camera & landmark overlay ───────────────────────────────────────────────

  async initCamera() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60 } },
        audio: false,
      });
      if (this.localVideo) this.localVideo.srcObject = this.stream;
      this._startLandmarkOverlay();
    } catch (err) {
      console.warn("Camera unavailable:", err);
      this._setCslrBadge("sim");
    }
  }

  _startLandmarkOverlay() {
    if (!this.landmarkCanvas || !this.localVideo) return;
    const ctx = this.landmarkCanvas.getContext("2d");
    const NODE_COLORS = {
      pose:  "rgba(139, 92, 246, 0.75)",   // violet
      rhand: "rgba(16, 185, 129, 0.85)",   // emerald
      lhand: "rgba(6, 182, 212, 0.85)",    // cyan
      face:  "rgba(245, 158, 11, 0.65)",   // amber
    };

    const drawLoop = () => {
      if (this.localVideo.readyState >= 2) {
        const W = this.localVideo.videoWidth  || 640;
        const H = this.localVideo.videoHeight || 480;
        this.landmarkCanvas.width  = W;
        this.landmarkCanvas.height = H;
        ctx.clearRect(0, 0, W, H);

        const t = Date.now() * 0.0025;
        const cx = W * 0.5, cy = H * 0.5;

        // Simulated hand joint rings (right hand — emerald)
        for (let i = 0; i < 5; i++) {
          const angle = t + (i * Math.PI * 2) / 5;
          const rx = cx + 90 + Math.sin(angle) * 22;
          const ry = cy + 60 + Math.cos(angle * 1.3) * 14;
          ctx.strokeStyle = NODE_COLORS.rhand;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(rx, ry, 5, 0, Math.PI * 2);
          ctx.stroke();
          ctx.fillStyle = NODE_COLORS.rhand;
          ctx.beginPath();
          ctx.arc(rx, ry, 2.5, 0, Math.PI * 2);
          ctx.fill();
        }

        // Left hand — cyan
        for (let i = 0; i < 5; i++) {
          const angle = -t + (i * Math.PI * 2) / 5;
          const lx = cx - 90 + Math.sin(angle) * 22;
          const ly = cy + 60 + Math.cos(angle * 1.3) * 14;
          ctx.strokeStyle = NODE_COLORS.lhand;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(lx, ly, 5, 0, Math.PI * 2);
          ctx.stroke();
          ctx.fillStyle = NODE_COLORS.lhand;
          ctx.beginPath();
          ctx.arc(lx, ly, 2.5, 0, Math.PI * 2);
          ctx.fill();
        }

        // Pose spine line — violet
        ctx.strokeStyle = NODE_COLORS.pose;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(cx - 50 + Math.sin(t * 0.5) * 5, cy - 80);
        ctx.lineTo(cx + 50 + Math.sin(t * 0.5) * 5, cy - 80);
        ctx.moveTo(cx, cy - 80);
        ctx.lineTo(cx, cy + 50);
        ctx.stroke();
        ctx.setLineDash([]);

        // Face contour ellipse — amber
        ctx.strokeStyle = NODE_COLORS.face;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.ellipse(cx, cy - 110 + Math.sin(t * 0.3) * 3, 28, 36, 0, 0, Math.PI * 2);
        ctx.stroke();
      }
      requestAnimationFrame(drawLoop);
    };
    requestAnimationFrame(drawLoop);
  }

  // ─── WebSocket — full-duplex room with auto-reconnect ───────────────────────

  initWebSocket() {
    if (this._wsReconnectTimer) clearTimeout(this._wsReconnectTimer);
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host   = window.location.host || "localhost:8000";
    const wsUrl  = `${proto}//${host}/ws/room/${this.config.roomId}/${this.config.clientType}`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this._setConnBadge("live");
        this._wsReconnectDelay = 1500;
        console.log("[WS] Connected to room:", wsUrl);
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "cslr_prediction" || data.gloss || data.text) {
            this._onCslrPrediction(data);
          } else if (data.type === "speaker_text") {
            this._onSpeakerText(data.text);
          }
        } catch (e) {
          console.warn("[WS] Parse error:", e);
        }
      };

      this.ws.onclose = () => {
        this._setConnBadge("reconnecting");
        this._scheduleReconnect();
      };

      this.ws.onerror = (err) => {
        console.warn("[WS] Error:", err);
        this._setConnBadge("error");
      };

    } catch (e) {
      console.warn("[WS] Init error:", e);
      this._setConnBadge("error");
    }
  }

  _scheduleReconnect() {
    const delay = Math.min(this._wsReconnectDelay, 15000);
    this._wsReconnectDelay = Math.floor(delay * 1.5);
    console.log(`[WS] Reconnecting in ${delay}ms…`);
    this._wsReconnectTimer = setTimeout(() => this.initWebSocket(), delay);
  }

  _startPingLoop() {
    this._pingInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: "ping" }));
      }
    }, 25000);
  }

  // ─── Signer → Speaker prediction handler ────────────────────────────────────

  _onCslrPrediction(data) {
    const text = data.text || data.gloss || "";
    if (!text) return;
    const conf = data.confidence !== undefined ? Math.round(data.confidence * 100) : "--";
    const lat  = data.latency_ms !== undefined ? `${data.latency_ms.toFixed(1)}ms` : "--";

    if (this.signerSubtitle) {
      this.signerSubtitle.innerText = text;
      this.signerSubtitle.classList.add("subtitle-flash");
      setTimeout(() => this.signerSubtitle.classList.remove("subtitle-flash"), 400);
    }
    if (this.confTag)   this.confTag.innerText   = `${conf}% নির্ভুল`;
    if (this.latencyTag) this.latencyTag.innerText = `⚡ ${lat}`;

    this._setCslrBadge("active");

    this.currentText = text;
    this._appendTranscriptRow("Signer 🤟", text);
    this._enqueueTTS(text);
  }

  // ─── Speaker → Signer handler ───────────────────────────────────────────────

  _onSpeakerText(text) {
    if (!text) return;
    if (this.avatar && typeof this.avatar.requestSignStream === "function") {
      this.avatar.requestSignStream(text);
    }
    this._appendTranscriptRow("Speaker 🗣️", text);
  }

  // ─── Speech recognition ─────────────────────────────────────────────────────

  initSpeechRecognition() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;
    this.recognition = new SR();
    this.recognition.lang = "bn-BD";
    this.recognition.continuous = false;
    this.recognition.interimResults = false;

    this.recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      if (this.speakerInput) this.speakerInput.value = transcript;
      this.sendSpeakerText(transcript);
      this.stopVoiceVisualizer();
    };

    this.recognition.onerror = () => this.stopVoiceVisualizer();
    this.recognition.onend   = () => this.stopVoiceVisualizer();
  }

  // ─── Voice waveform visualizer ──────────────────────────────────────────────

  startVoiceVisualizer() {
    if (!this.waveformCanvas) return;
    this.isRecording = true;
    if (this.btnMic) this.btnMic.classList.add("recording");

    const ctx   = this.waveformCanvas.getContext("2d");
    const barCount = 28;
    const barHeights = Array(barCount).fill(0).map(() => Math.random() * 28 + 4);

    const drawWave = () => {
      if (!this.isRecording) {
        ctx.clearRect(0, 0, this.waveformCanvas.width, this.waveformCanvas.height);
        return;
      }
      this.waveformCanvas.width  = this.waveformCanvas.clientWidth || 320;
      this.waveformCanvas.height = 44;
      ctx.clearRect(0, 0, this.waveformCanvas.width, this.waveformCanvas.height);

      const barW = this.waveformCanvas.width / barCount;
      for (let i = 0; i < barCount; i++) {
        // Smooth random walk
        barHeights[i] = Math.max(4, Math.min(38, barHeights[i] + (Math.random() - 0.5) * 8));
        const h = barHeights[i];
        const alpha = 0.5 + (h / 38) * 0.5;
        ctx.fillStyle = `rgba(6, 182, 212, ${alpha.toFixed(2)})`;
        ctx.beginPath();
        ctx.roundRect(i * barW + 2, (44 - h) / 2, barW - 4, h, 3);
        ctx.fill();
      }
      requestAnimationFrame(drawWave);
    };
    requestAnimationFrame(drawWave);
  }

  stopVoiceVisualizer() {
    this.isRecording = false;
    if (this.btnMic) this.btnMic.classList.remove("recording");
  }

  // ─── Speaker text send ───────────────────────────────────────────────────────

  sendSpeakerText(text) {
    text = (text || "").trim();
    if (!text) return;

    this._onSpeakerText(text);

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "speaker_text", text }));
    }

    if (this.speakerInput) this.speakerInput.value = "";
  }

  // ─── Bengali TTS queue (prevents overlapping utterances) ────────────────────

  _enqueueTTS(text) {
    if (!text || !window.speechSynthesis) return;
    this._ttsQueue.push(text);
    if (!this._ttsBusy) this._drainTTS();
  }

  _drainTTS() {
    if (this._ttsQueue.length === 0) {
      this._ttsBusy = false;
      return;
    }
    this._ttsBusy = true;
    const text = this._ttsQueue.shift();
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang  = "bn-BD";
    utt.rate  = 0.92;
    utt.pitch = 1.0;
    utt.onend = () => setTimeout(() => this._drainTTS(), 100);
    utt.onerror = () => this._drainTTS();
    window.speechSynthesis.speak(utt);
  }

  vocalizeBangla(text) {
    this._enqueueTTS(text);
  }

  // ─── Transcript panel ───────────────────────────────────────────────────────

  _appendTranscriptRow(speaker, text) {
    const entry = {
      speaker,
      text,
      timestamp: new Date().toLocaleTimeString("bn-BD"),
    };
    this.transcriptLogs.push(entry);

    if (!this.transcriptList) return;
    const li = document.createElement("li");
    li.className = "transcript-row";
    li.innerHTML = `
      <span class="tr-ts">${entry.timestamp}</span>
      <span class="tr-speaker">${speaker}</span>
      <span class="tr-text">${this._escapeHTML(text)}</span>
    `;
    this.transcriptList.prepend(li);
    if (this.transcriptList.children.length > 40) {
      this.transcriptList.removeChild(this.transcriptList.lastChild);
    }
  }

  _escapeHTML(str) {
    return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  }

  // ─── Badge helpers ───────────────────────────────────────────────────────────

  _setConnBadge(state) {
    if (!this.connBadge) return;
    const COLORS = { live: "#10b981", reconnecting: "#f59e0b", error: "#ef4444" };
    const LABELS = { live: "● লাইভ", reconnecting: "↺ সংযোগ…", error: "✕ বিচ্ছিন্ন" };
    this.connBadge.style.color = COLORS[state] || "#94a3b8";
    this.connBadge.innerText   = LABELS[state] || state;
  }

  _setCslrBadge(state) {
    if (!this.cslrFeedBadge) return;
    if (state === "active") {
      this.cslrFeedBadge.innerText = "● CSLR সক্রিয়";
      this.cslrFeedBadge.style.color = "#10b981";
    } else if (state === "sim") {
      this.cslrFeedBadge.innerText = "◌ সিমুলেশন মোড";
      this.cslrFeedBadge.style.color = "#f59e0b";
    }
  }

  // ─── Event listeners ─────────────────────────────────────────────────────────

  initEventListeners() {
    if (this.btnSend && this.speakerInput) {
      this.btnSend.addEventListener("click", () => {
        this.sendSpeakerText(this.speakerInput.value);
      });
      this.speakerInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") this.sendSpeakerText(this.speakerInput.value);
      });
    }

    if (this.btnMic) {
      this.btnMic.addEventListener("click", () => {
        if (this.recognition) {
          this.startVoiceVisualizer();
          this.recognition.start();
        } else {
          alert("এই ব্রাউজারে স্পিচ রিকগনিশন সমর্থিত নয়।");
        }
      });
    }

    if (this.btnTTS) {
      this.btnTTS.addEventListener("click", () => {
        if (this.currentText) this.vocalizeBangla(this.currentText);
      });
    }

    if (this.btnExport) {
      this.btnExport.addEventListener("click", () => {
        const blob = new Blob(
          [JSON.stringify(this.transcriptLogs, null, 2)],
          { type: "application/json" }
        );
        const url = URL.createObjectURL(blob);
        const a   = document.createElement("a");
        a.href     = url;
        a.download = `IsharaConnect_Transcript_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
      });
    }

    if (this.btnToggleCam && this.stream) {
      this.btnToggleCam.addEventListener("click", () => {
        const vt = this.stream.getVideoTracks()[0];
        if (vt) {
          vt.enabled = !vt.enabled;
          this.btnToggleCam.innerText = vt.enabled
            ? "📷 ক্যামেরা বন্ধ করুন" : "📷 ক্যামেরা চালু করুন";
        }
      });
    }

    // Cleanup on page unload
    window.addEventListener("beforeunload", () => {
      if (this._pingInterval) clearInterval(this._pingInterval);
      if (this._wsReconnectTimer) clearTimeout(this._wsReconnectTimer);
      if (this.ws) this.ws.close();
    });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.dashboardController = new DashboardController();
});
