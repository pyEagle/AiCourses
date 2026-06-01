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
    
    def check(self, input_text):
        if not self.flag: # 未被唤醒
            if input_text != self.WAKE_WORD:
                return (False, 0)
    
            self.date_timer = self.get_current_time()
            self.flag = True

            return (True, 1) # 初次被唤醒
        else: # 已被唤醒
            temp_time = self.get_current_time()
            time_diff = temp_time - self.date_timer
            if time_diff < self.MAX_INTERVAL:
                self.date_timer = self.get_current_time()
                
                return (True, 2) # 有效交流状态
            else: # 设置为唤醒状态
                self.flag = False
                self.date_timer = None
                return (False, 0) 
