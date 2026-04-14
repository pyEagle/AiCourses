# -*- coding:utf-8 -*-

import cv2 as cv
import tkinter as tk


def get_screen_resolution():
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    
    return screen_width, screen_height

def resize_image(image, screen_w, screen_h, scale_ratio=0.5):
        img_h, img_w = image.shape[:2] # image.shape: [h, w, c]
        scale_w = (screen_w * scale_ratio)/img_w
        scale_h = (screen_h * scale_ratio)/img_h
        scale = min(scale_w, scale_h)

        if scale < 1:
            resized_img = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        else:
            resized_img = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

        return resized_img

