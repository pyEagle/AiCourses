# -*- coding: utf-8 -*-

import os
import json
import logging

import tornado
from tornado.web import RequestHandler, Application
from tornado.options import define, options
from paddlespeech.cli.asr.infer import ASRExecutor

define("port", default=6060, help="运行端口", type=int) 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

asr_executor = ASRExecutor()

class ASRHandler(RequestHandler):
    async def post(self):
        try:
            audio_file = self.request.files.get('audio')[0]['body']
            result = asr_executor(audio_file=audio_file)
            
            self.set_header("Content-Type", "application/json")
            self.write(json.dumps({"text": result}, ensure_ascii=False))
        except Exception as e:
            logger.error(f"ASR 处理失败: {str(e)}", exc_info=True)
            self.set_status(500)
            self.write(json.dumps({"error": f"识别失败: {str(e)}"}))

def make_app():
    return Application([(r"/asr", ASRHandler)], autoreload=True)

if __name__ == "__main__":
    options.parse_command_line()
    app = make_app()
    app.listen(options.port, address="0.0.0.0")
    print(f"Speech API 服务已启动，监听端口: {options.port}")
    tornado.ioloop.IOLoop.current().start()
