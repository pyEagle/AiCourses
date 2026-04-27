# -*- coding:utf-8 -*-

import os
import uuid
import asyncio

from core.agent import Agent

from dotenv import load_dotenv
from aibot import WSClient, WSClientOptions, generate_req_id

load_dotenv()

agent = Agent()
session_id = str(uuid.uuid4())

ws_client = WSClient(
    WSClientOptions(
        bot_id=os.getenv('WECHAT_BOT_ID', 'your_WECHAT_BOT_ID'),
        secret=os.getenv('WECHAT_BOT_SECRET', 'your_WECHAT_BOT_SECRET'),
    )
)

@ws_client.on('authenticated')
def on_authenticated():
    print('🔐 认证成功')

@ws_client.on('message.text')
async def on_text(frame):
    content = frame.get('body', {}).get('text', {}).get('content', '')
    print(f'收到文本: {content}')

    stream_id = generate_req_id('stream')

    await ws_client.reply_stream(frame, stream_id, '正在思考中...', False)

    try:
        answer = agent.chat(session_id, content)
    except Exception as e:
        import traceback
        traceback.print_exc()
        answer = "我在回去修炼一段时间，才能回答你的问题"

    await ws_client.reply_stream(frame, stream_id, f'{answer}', True)

@ws_client.on('event.enter_chat')
async def on_enter_chat(frame):
    await ws_client.reply_welcome(frame, {
        'msgtype': 'text',
        'text': {'content': '您好！我是智能助手，有什么可以帮您的吗？'},
    })

ws_client.run()


