/**
 * IsharaConnect - Real-time Two-Way Split Dashboard Controller
 * Connects WebRTC camera feed, CSLR subtitles, Voice STT, 3D Avatar, and WebSocket room.
 */

class DashboardController {
  constructor() {
    this.config = window.BDSL_CONFIG || { roomId: "room-general", clientType: "signer" };
    this.transcriptLogs = [];
    this.currentText = "";
    this.isRecording = false;

    this.initDOMElements();
    this.initAvatar();
    this.initCamera();
    this.initWebSocket();
    this.initSpeechRecognition();
    this.initEventListeners();
  }

  initDOMElements() {
    this.localVideo = document.getElementById("local-video");
    this.landmarkCanvas = document.getElementById("landmark-canvas");
    this.signerSubtitle = document.getElementById("signer-subtitle");
    this.confTag = document.getElementById("confidence-tag");
    this.speakerInput = document.getElementById("speaker-text-input");
    this.btnSend = document.getElementById("btn-send-synthesize");
    this.btnMic = document.getElementById("btn-voice-stt");
    this.btnTTS = document.getElementById("btn-tts-speak");
    this.btnExport = document.getElementById("btn-export-transcript");
    this.btnToggleCam = document.getElementById("btn-toggle-cam");
    this.roleSelect = document.getElementById("role-select");
    this.connBadge = document.getElementById("conn-badge");
    this.waveformCanvas = document.getElementById("audio-waveform");
  }

  initAvatar() {
    if (typeof HumanoidBdSLAvatar !== "undefined") {
      this.avatar = new HumanoidBdSLAvatar("avatar-container");
    } else if (typeof Ishara3DFACSAvatar !== "undefined") {
      this.avatar = new Ishara3DFACSAvatar("avatar-container");
    }
  }

  async initCamera() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60 } },
        audio: false
      });
      if (this.localVideo) {
        this.localVideo.srcObject = this.stream;
      }
      this.startLandmarkOverlayLoop();
    } catch (err) {
      console.warn("Camera access denied or unavailable:", err);
    }
  }

  startLandmarkOverlayLoop() {
    if (!this.landmarkCanvas || !this.localVideo) return;
    const ctx = this.landmarkCanvas.getContext("2d");

    const drawLoop = () => {
      if (this.localVideo.readyState >= 2) {
        this.landmarkCanvas.width = this.localVideo.videoWidth || 640;
        this.landmarkCanvas.height = this.localVideo.videoHeight || 480;
        ctx.clearRect(0, 0, this.landmarkCanvas.width, this.landmarkCanvas.height);

        // Draw dynamic tracking node markers
        const t = Date.now() * 0.003;
        const w = this.landmarkCanvas.width;
        const h = this.landmarkCanvas.height;

        // Simulated central wrist & hand joints
        const wx = w * 0.5 + Math.sin(t) * 20;
        const wy = h * 0.65 + Math.cos(t) * 15;

        ctx.strokeStyle = "rgba(6, 182, 212, 0.6)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(wx, wy, 8, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = "#10b981";
        ctx.beginPath();
        ctx.arc(wx, wy, 4, 0, Math.PI * 2);
        ctx.fill();
      }
      requestAnimationFrame(drawLoop);
    };
    requestAnimationFrame(drawLoop);
  }

  initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host || "localhost:8000";
    const wsUrl = `${protocol}//${host}/ws/room/${this.config.roomId}/${this.config.clientType}`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        if (this.connBadge) {
          this.connBadge.style.color = "#10b981";
        }
      };

      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "cslr_prediction" || data.text) {
          this.onSignerPredictionReceived(data);
        } else if (data.type === "speaker_text" && this.avatar) {
          this.avatar.requestSignStream(data.text);
        }
      };

      this.ws.onclose = () => {
        if (this.connBadge) {
          this.connBadge.style.color = "#f59e0b";
        }
      };
    } catch (e) {
      console.warn("WebSocket initialization error:", e);
    }
  }

  onSignerPredictionReceived(data) {
    const text = data.text || data.gloss || "";
    const conf = data.confidence !== undefined ? Math.round(data.confidence * 100) : 95;

    if (this.signerSubtitle) {
      this.signerSubtitle.innerText = text;
    }
    if (this.confTag) {
      this.confTag.innerText = `${conf}% নির্ভুল`;
    }

    this.currentText = text;
    this.transcriptLogs.push({
      speaker: "Signer",
      text: text,
      timestamp: new Date().toLocaleTimeString("bn-BD")
    });

    // Auto vocalize if enabled
    this.vocalizeBangla(text);
  }

  initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      this.recognition = new SpeechRecognition();
      this.recognition.lang = "bn-BD";
      this.recognition.continuous = false;
      this.recognition.interimResults = false;

      this.recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        if (this.speakerInput) {
          this.speakerInput.value = transcript;
        }
        this.sendSpeakerText(transcript);
        this.stopVoiceVisualizer();
      };

      this.recognition.onerror = () => {
        this.stopVoiceVisualizer();
      };

      this.recognition.onend = () => {
        this.stopVoiceVisualizer();
      };
    }
  }

  startVoiceVisualizer() {
    if (!this.waveformCanvas) return;
    this.isRecording = true;
    if (this.btnMic) this.btnMic.classList.add("recording");

    const ctx = this.waveformCanvas.getContext("2d");
    const drawWave = () => {
      if (!this.isRecording) {
        ctx.clearRect(0, 0, this.waveformCanvas.width, this.waveformCanvas.height);
        return;
      }
      this.waveformCanvas.width = this.waveformCanvas.clientWidth || 300;
      this.waveformCanvas.height = 40;
      ctx.clearRect(0, 0, this.waveformCanvas.width, this.waveformCanvas.height);

      ctx.fillStyle = "rgba(6, 182, 212, 0.8)";
      const bars = 24;
      const barW = this.waveformCanvas.width / bars;
      for (let i = 0; i < bars; i++) {
        const h = Math.random() * 30 + 5;
        ctx.fillRect(i * barW + 2, (40 - h) / 2, barW - 4, h);
      }
      requestAnimationFrame(drawWave);
    };
    requestAnimationFrame(drawWave);
  }

  stopVoiceVisualizer() {
    this.isRecording = false;
    if (this.btnMic) this.btnMic.classList.remove("recording");
  }

  sendSpeakerText(text) {
    if (!text) return;
    if (this.avatar && typeof this.avatar.requestSignStream === "function") {
      this.avatar.requestSignStream(text);
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "speaker_text", text: text }));
    }

    this.transcriptLogs.push({
      speaker: "Speaker",
      text: text,
      timestamp: new Date().toLocaleTimeString("bn-BD")
    });
  }

  vocalizeBangla(text) {
    if (!text || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "bn-BD";
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
  }

  initEventListeners() {
    if (this.btnSend && this.speakerInput) {
      this.btnSend.addEventListener("click", () => {
        const txt = this.speakerInput.value.trim();
        if (txt) this.sendSpeakerText(txt);
      });

      this.speakerInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
          const txt = this.speakerInput.value.trim();
          if (txt) this.sendSpeakerText(txt);
        }
      });
    }

    if (this.btnMic) {
      this.btnMic.addEventListener("click", () => {
        if (this.recognition) {
          this.startVoiceVisualizer();
          this.recognition.start();
        } else {
          alert("আপনার ব্রাউজারে স্পিচ রিকগনিশন সমর্থিত নয়। দয়া করে টাইপ করুন।");
        }
      });
    }

    if (this.btnTTS) {
      this.btnTTS.addEventListener("click", () => {
        if (this.currentText) {
          this.vocalizeBangla(this.currentText);
        }
      });
    }

    if (this.btnExport) {
      this.btnExport.addEventListener("click", () => {
        const blob = new Blob([JSON.stringify(this.transcriptLogs, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `IsharaConnect_Transcript_${Date.now()}.json`;
        a.click();
      });
    }

    if (this.btnToggleCam && this.stream) {
      this.btnToggleCam.addEventListener("click", () => {
        const videoTrack = this.stream.getVideoTracks()[0];
        if (videoTrack) {
          videoTrack.enabled = !videoTrack.enabled;
          this.btnToggleCam.innerText = videoTrack.enabled ? "📷 ক্যামেরা বন্ধ করুন" : "📷 ক্যামেরা চালু করুন";
        }
      });
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.dashboardController = new DashboardController();
});
