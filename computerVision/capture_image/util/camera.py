# -*- coding:utf-8 -*-

import cv2 as cv

class Camera(object):
    @staticmethod
    def get_available_cameras(max_index=6):
        for idx in range(max_index):
            cap = cv.VideoCapture(idx)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"成功连接到摄像头索引 {idx}")
                    return cap
                else:
                    cap.release()
                    print(f"摄像头索引 {idx} 已打开但无法读取帧，跳过")
            else:
                print(f"摄像头索引 {idx} 无法打开，跳过")

        return None

    def init(self):
        print("正在检测可用摄像头...")
        return self.get_available_cameras()


