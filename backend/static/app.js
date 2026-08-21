document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('start-btn');
    const videoOverlay = document.getElementById('video-overlay');
    const localVideo = document.getElementById('local-video');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const outputBox = document.getElementById('output-box');
    const glossOutput = document.getElementById('gloss-output');
    const speakBtn = document.getElementById('speak-btn');
    
    let currentTranslation = "";

    startBtn.addEventListener('click', async () => {
        try {
            // UI Update
            startBtn.innerHTML = '<span class="btn-icon">⌛</span> Connecting...';
            startBtn.disabled = true;

            // Initialize WebRTC
            const pc = new RTCPeerConnection({
                iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
            });

            // লাইভ রেজাল্ট পাওয়ার জন্য DataChannel
            const dataChannel = pc.createDataChannel("ishara_results");
            dataChannel.onmessage = (event) => {
                try {
                    const result = JSON.parse(event.data.replace(/'/g, '"'));
                    console.log("Gloss:", result.gloss);
                    console.log("Bengali Text:", result.text);
                    
                    glossOutput.innerText = result.gloss || "-";
                    
                    if (result.text && result.text !== currentTranslation) {
                        currentTranslation = result.text;
                        outputBox.innerHTML = `<p>${result.text}</p>`;
                        
                        // Add a slight flash effect for new translation
                        outputBox.style.transform = 'scale(1.02)';
                        setTimeout(() => outputBox.style.transform = 'scale(1)', 200);
                    }
                } catch (e) {
                    console.error("Error parsing message", e);
                }
            };

            pc.onconnectionstatechange = () => {
                if (pc.connectionState === 'connected') {
                    statusDot.classList.remove('disconnected');
                    statusDot.classList.add('connected');
                    statusText.innerText = 'Connected';
                    videoOverlay.classList.add('hidden');
                    outputBox.innerHTML = '<p class="placeholder-text">Sign into the camera...</p>';
                } else if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
                    statusDot.classList.remove('connected');
                    statusDot.classList.add('disconnected');
                    statusText.innerText = 'Disconnected';
                    videoOverlay.classList.remove('hidden');
                    startBtn.innerHTML = '<span class="btn-icon">▶</span> Reconnect';
                    startBtn.disabled = false;
                }
            };

            // ক্যামেরা স্ট্রিম যুক্ত করা
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, frameRate: 30 } });
            localVideo.srcObject = stream;
            
            stream.getTracks().forEach(track => pc.addTrack(track, stream));
            
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            
            // FastAPI /offer এন্ডপয়েন্টে SDP পাঠানো
            const response = await fetch("http://localhost:8000/offer", {
                method: "POST",
                body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
                headers: { "Content-Type": "application/json" }
            });
            
            const answer = await response.json();
            await pc.setRemoteDescription(new RTCSessionDescription(answer));

        } catch (error) {
            console.error("Error starting camera or WebRTC:", error);
            startBtn.innerHTML = '<span class="btn-icon">❌</span> Error. Try Again';
            startBtn.disabled = false;
            alert("Could not access camera or connect to server. Ensure camera permissions are granted and backend is running.");
        }
    });

    // Text to Speech
    speakBtn.addEventListener('click', () => {
        if (currentTranslation && 'speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(currentTranslation);
            utterance.lang = 'bn-BD'; // Bengali language
            window.speechSynthesis.speak(utterance);
        }
    });
});
