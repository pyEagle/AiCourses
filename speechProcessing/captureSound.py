# -*- coding:utf-8 -*- 

import pyaudio

class CaptureSound:
    def __init__(self):
        self.FORMAT = pyaudio.paInt16  # 16位采样
        self.CHANNELS = 1              # 单声道
        self.RATE = 16000              # 采样率 16k
        self.CHUNK = 1024              # 每次读取的帧数

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format = self.FORMAT,
            channels = self.CHANNELS,
            rate = self.RATE,
            input = True,
            frames_per_buffer = self.CHUNK,
        )
        
        print("实时语音采集已启动，按 Ctrl+C 停止...")

        try:
            while True:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                yield data
        except KeyboardInterrupt:
            print("用户手动停止采集")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

            print("采集设备已安全关闭")

if __name__ == "__main__":
    cs = CaptureSound()
    audio_stream = cs.run()
    try:
        for chunk_data in audio_stream:
            print(f"成功采集到音频片段，大小: {len(chunk_data)} 字节")
    except KeyboardInterrupt:
        pass

