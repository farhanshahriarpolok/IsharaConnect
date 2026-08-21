"""WebRTC Track Processor for Sign Language Recognition."""

import asyncio
import json
import numpy as np
from aiortc import MediaStreamTrack
from av import VideoFrame

from core_engine.inference.cslr_engine import SlidingWindowBuffer, IsharaInferenceEngine

class SignLanguageTrackProcessor(MediaStreamTrack):
    kind = "video"

    def __init__(self, track: MediaStreamTrack, data_channel, engine: IsharaInferenceEngine):
        super().__init__()
        self.track = track
        self.data_channel = data_channel
        self.engine = engine
        self.buffer = SlidingWindowBuffer(window_size=32, stride=8)
        self.last_gloss = ""

    async def recv(self):
        # রিয়েল-টাইম ভিডিও ফ্রেম গ্রহণ
        frame: VideoFrame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # ১. লাইটওয়েট ফিচার / ল্যান্ডমার্ক রিপ্রেজেন্টেশন এক্সট্রাকশন (Dummy Shape: 75, 3)
        # প্রোডাকশনে: MediaPipe C++ বাইন্ডিং বা TensorRT ল্যান্ডমার্ক এক্সট্রাক্টর
        mock_landmarks = np.zeros((75, 3), dtype=np.float32)

        # ২. স্লাইডিং উইন্ডো আপডেট ও ইনফারেন্স
        if self.buffer.append(mock_landmarks):
            window = self.buffer.get_window()
            
            # অ্যাসিনক্রোনাস নন-ব্লকিং ইনফারেন্স কল
            asyncio.create_task(self._process_inference(window))

        return frame

    async def _process_inference(self, window: np.ndarray):
        gloss = await self.engine.predict_cslr_ctc(window)
        
        # ডুপ্লিকেট সিকোয়েন্স ফিল্টারিং
        if gloss and gloss != self.last_gloss:
            self.last_gloss = gloss
            translated_text = await self.engine.translate_gloss_to_text(gloss)

            # WebRTC DataChannel-এর মাধ্যমে রেজাল্ট ফ্রন্টএন্ডে পুশ
            if self.data_channel and self.data_channel.readyState == "open":
                payload = {
                    "gloss": gloss,
                    "text": translated_text
                }
                self.data_channel.send(json.dumps(payload))
