# -*- coding:utf-8 -*- 

import pyaudio

class CaptureSound:
    def __init__(self, input_device_index=None):
        self.FORMAT = pyaudio.paInt16  # 16位采样
        self.CHANNELS = 1              # 单声道
        self.RATE = 16000              # 采样率 16k
        self.CHUNK = 1024              # 每次读取的帧数
        self.input_device_index = input_device_index  # 可以指定声卡

    def list_audio_devices(self):
        p = pyaudio.PyAudio()
        print("\n======= 所有声卡/录音设备列表 =======")
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0:
                print(f"设备编号: {i} | 名称: {dev['name']}")
        p.terminate()

    def run(self):
        p = pyaudio.PyAudio()

        if self.input_device_index is None:
            device_info = p.get_default_input_device_info()
            print(f"\n当前使用：【系统默认声卡】")
        else:
            device_info = p.get_device_info_by_index(self.input_device_index)
            print(f"\n当前使用：【指定声卡】")
            
        print(f"声卡编号: {device_info['index']}")
        print(f"声卡名称: {device_info['name']}")

        # 打开音频流
        stream = p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=self.input_device_index,
        )
        
        print("\n实时语音采集已启动，按 Ctrl+C 停止...")

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
