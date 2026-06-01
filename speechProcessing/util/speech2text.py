# -*- coding: utf-8 -*-

import requests

def asr_text(audio_path):
    url = "http://127.0.0.1:8090/asr"

    result = ''
    try:
        files = {"audio": audio_path}
        response = requests.post(url, files=files)
        result = response.json()['text']
    except Exception as e:
        print(f"请求失败：{str(e)}")

    return result

