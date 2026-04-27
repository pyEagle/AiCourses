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

# 1. 创建客户端实例
ws_client = WSClient(
    WSClientOptions(
        bot_id=os.getenv('WECHAT_BOT_ID', 'aib8kHIvLrO0tizx4SmBrb7_9jVzrq28Onw'),
        secret=os.getenv('WECHAT_BOT_SECRET', 'EGxaEM5GlJ3FKvqCjO5ca1Qeo6fdDHpBpXSHBJeuIy3'),
    )
)

# 2. 监听认证成功
@ws_client.on('authenticated')
def on_authenticated():
    print('🔐 认证成功')

# 3. 监听文本消息并进行流式回复
@ws_client.on('message.text')
async def on_text(frame):
    content = frame.get('body', {}).get('text', {}).get('content', '')
    print(f'收到文本: {content}')

    stream_id = generate_req_id('stream')

    await ws_client.reply_stream(frame, stream_id, '正在思考中...', False)

    # 发送最终结果
    try:
        answer = agent.chat(session_id, content)
    except Exception as e:
        import traceback
        traceback.print_exc()
        answer = "我在回去修炼一段时间，才能回答你的问题"

    await ws_client.reply_stream(frame, stream_id, f'{answer}', True)

# 4. 监听进入会话事件（发送欢迎语）
@ws_client.on('event.enter_chat')
async def on_enter_chat(frame):
    await ws_client.reply_welcome(frame, {
        'msgtype': 'text',
        'text': {'content': '您好！我是智能助手，有什么可以帮您的吗？'},
    })

# 5. 启动（便捷方法，内部管理事件循环）
ws_client.run()


