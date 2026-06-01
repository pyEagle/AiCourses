# -*- coding:utf-8 -*-
import os
import tempfile
import wave
import time

import tornado.ioloop
import tornado.web
import tornado.gen
import requests

from paddlespeech.cli.asr.infer import ASRExecutor

from util.edge_intent import EdgeIntentEngine
from util.voice_wake_up import VoiceWakeUp
from util.speech2text import asr_text
from util.capture_sound import CaptureSound
from config.settings import VOICE

TEMP_DIR = './data/sound/'
os.makedirs(TEMP_DIR, exist_ok=True)


class LocalAsr:
    def __init__(self):
        self.cap_sound = CaptureSound()
        self.vwp = VoiceWakeUp(VOICE['WAPEUPWORD'])
        self.intent = EdgeIntentEngine(confidence_threshold=0.45, similarity_threshold=0.7)
        self.intent.load()

        self.asr_executor = ASRExecutor()
        self.url = ''

        self.audio_buffer = bytes()  # 音频缓存
        self.SAMPLE_RATE = 16000     # 必须和录音一致
        self.CHANNELS = 1
        self.SAMPLE_WIDTH = 2
        self.MIN_SECONDS = 0.5       # 最小识别时长：0.5秒
        self.MIN_BYTES = int(self.SAMPLE_RATE * self.SAMPLE_WIDTH * self.CHANNELS * self.MIN_SECONDS)

    def run(self):
        for chunk in self.cap_sound.run():
            js_data = {}
            temp_wav = ""
            
            try:
                self.audio_buffer += chunk
                current_bytes = len(self.audio_buffer)
                print(f"缓存大小：{current_bytes} / 需要：{self.MIN_BYTES}")

                if current_bytes < self.MIN_BYTES:
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", dir=TEMP_DIR, delete=False) as f:
                    temp_wav = f.name

                with wave.open(temp_wav, 'wb') as wav_file:
                    wav_file.setnchannels(self.CHANNELS)
                    wav_file.setsampwidth(self.SAMPLE_WIDTH)
                    wav_file.setframerate(self.SAMPLE_RATE)
                    wav_file.writeframes(self.audio_buffer)

                text = self.asr_executor(audio_file=temp_wav)
                input_text = text.strip()
                print('识别结果：', input_text)

                self.audio_buffer = bytes()

                api_result = ""
                check = self.vwp.check(input_text)
                if check[0]:
                    if check[1] == 1:
                        api_result = VOICE['FIRST_ANSWER']
                        print('唤醒成功：', VOICE['FIRST_ANSWER'])
                    else:
                        api_result = self.intent.predict(input_text)
                        print('意图结果：', api_result)

                js_data = {
                    "code": 200,
                    "msg": "success",
                    "data": api_result,
                    "text": input_text
                }

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.audio_buffer = bytes()
                js_data = {
                    "code": 500,
                    "msg": f"error: {str(e)}",
                    "data": ""
                }

            finally:
                if temp_wav and os.path.exists(temp_wav):
                    os.remove(temp_wav)

if __name__ == "__main__":
    local_asr = LocalAsr()
    local_asr.run()

