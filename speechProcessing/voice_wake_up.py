# -*- coding:utf-8 -*-

import time


class VoiceWakeUp:
    def __init__(self, wake_word, max_inter=10):
        self.WAKE_WORD = wake_word
        self.MAX_INTERVAL = max_inter

        self.flag = False
        self.date_timer = None
    
    def get_current_time(self):
        return time.time()
    
    def run(self):
        print("=== 语音唤醒已启动 ===")
        print(f"唤醒词：[{self.WAKE_WORD}]，超时时间：{self.MAX_INTERVAL}秒")
    
        while True:
            input_text = input("你说：").strip()
            if input_text == "exit":
                print("系统退出！")
                break
    
            if not input_text:
                continue
    
            if not self.flag: # 未被唤醒
                if input_text != self.WAKE_WORD:
                    continue
    
                print("AI：在的，可以帮你什么")
                self.date_timer = self.get_current_time()
                self.flag = True
            else: # 已被唤醒
                temp_time = self.get_current_time()
                time_diff = temp_time - self.date_timer
                if time_diff < self.MAX_INTERVAL: # 两句话，间隔不应该太长
                    print(f"AI：已收到指令 -> {input_text}")
                    self.date_timer = self.get_current_time()
                else:
                    self.flag = False
                    self.date_timer = None
                    print("AI：先回去睡个回笼觉嘞")

