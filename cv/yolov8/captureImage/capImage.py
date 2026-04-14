# -*- coding:utf-8 -*-

import os
import time
import datetime
import traceback

import numpy
import cv2 as cv

from util.camera import Camera
from util.guiEvent import mouse_click, handle_key_event
from util.screenResolution import get_screen_resolution, resize_image


class CollectImage(object):
    def __init__(self, save_image_dir='./data/image'):
        self.save_image_dir = save_image_dir
        if not os.path.exists(save_image_dir):
            os.makedirs(save_image_dir)

        self.cap = self.get_cap_handle()

    @staticmethod
    def get_cap_handle():
        cam = Camera()
        return cam.init()

    def run(self):
        window_name = "CollectImage"
        screen_w, screen_h = get_screen_resolution()
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

        try:
            while True:
                ret, current_frame = self.cap.read()
                if not ret or current_frame is None:
                    time.sleep(0.001)
                    continue

                fframe = resize_image(current_frame, screen_w, screen_h)

                cv.imshow(window_name, fframe)

                key = cv.waitKey(1) & 0xFF
                image_file = self.gen_image_filename()
                if handle_key_event(key, fframe, image_file):
                    break
        finally:
            self.cap.release()
            cv.destroyAllWindows()

    def gen_image_filename(self):
        temp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        image_file = os.path.join(self.save_image_dir, f"image_{temp}.png")

        return image_file


if __name__ == "__main__":
    import sys

    save_image_dir = sys.argv[1]
    os.makedirs(save_image_dir, exist_ok=True)

    cm = CollectImage(save_image_dir)
    cm.run()

