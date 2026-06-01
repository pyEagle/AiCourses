# -*- coding:utf-8 -*-

import os
import datetime

import cv2 as cv

def mouse_click(event, x, y, flags, param):
    frame = param.get('frame')
    image_file = param.get('image_file')
    if event == cv.EVENT_LBUTTONDOWN and frame is not None:
        cv.imwrite(image_file, frame)
        print(f"图片已保存为 {image_filename}")

def handle_key_event(key, frame, image_file):
    if key == ord('s'): # s键保存
        cv.imwrite(image_file, frame)
        print(f"图片已保存为 {image_file}")
        return False 
    elif key == ord('q'):
        return True 

    return False
