# -*- coding:utf-8 -*-

import tornado.ioloop
import tornado.web
import tornado.gen

from util.edge_intent import EdgeIntentEngine
from util.voice_wake_up import VoiceWakeUp
from util.speech2text import asr_text
from config.settings import VOICE

vwp = VoiceWakeUp(VOICE['WAPEUPWORD'])
intent = EdgeIntentEngine(confidence_threshold=0.45,
                          similarity_threshold=0.7)
intent.load()

class IntentHandler(tornado.web.RequestHandler):
    @tornado.gen.coroutine
    def post(self):
        try:
            text = self.get_argument("text", "")
            audio_file = self.request.files.get("audio_file", None)

            input_text = text.strip()
            api_result = ""
            check = vwp.check(input_text)
            if check[0]:
                if check[1] == 1:
                    api_result = VOICE['FIRST_ANSWER']
                    print('FIRST_ANSWER:', VOICE['FIRST_ANSWER'])
                else:
                    api_result = intent.predict(input_text)
                    print('intent.predict: ', api_result)

            self.write({
                "code": 200,
                "msg": "success",
                "data": api_result,
                "text": input_text
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.write({
                "code": 500,
                "msg": f"error: {str(e)}",
                "data": ""
            })

def make_app():
    return tornado.web.Application([
        (r"/api/intent", IntentHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("Tornado 服务启动成功：http://127.0.0.1:8888/api/intent")
    tornado.ioloop.IOLoop.current().start()

