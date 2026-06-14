# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


def unsupervised_text_masking(image_path):
    img = cv2.imread(image_path)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # HSV色彩空间, 并提取 V 通道
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    # Otsu阈值分割
    ret_val, binary_mask = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"Otsu算法最佳分割阈值为: {ret_val}")

    # 形态学:闭运算 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    final_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    overlay_result = img_rgb.copy()
    overlay_result[final_mask == 255] = [0, 255, 0]

    plt.figure(figsize=(15, 5))
    plt.title("result")
    plt.imshow(overlay_result)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return final_mask, overlay_result

if __name__ == "__main__":
    import sys

    image_file = sys.argv[1]
    mask, green_overlay_img = unsupervised_text_masking(image_file)
