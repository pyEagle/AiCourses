# -*- coding:utf-8 -*-

import os
import datetime
import cv2 as cv

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


mouse_x, mouse_y = 0, 0
def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

cv.namedWindow('Camera')
cv.setMouseCallback('Camera', mouse_callback)

cap = get_available_cameras()
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        resolution_text = f"resolution: {int(width)}x{int(height)}"

        cv.putText(frame, resolution_text, (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        mouse_text = f"({mouse_x}, {mouse_y})"
        cv.putText(frame, mouse_text, (mouse_x + 10, mouse_y + 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv.imshow('Camera', frame)

        if cv.waitKey(1) == ord('q'):
            break

finally:
    cap.release()
    cv.destroyAllWindows()



