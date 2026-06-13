import cv2
import numpy as np

def unsupervised_character_detection(image_path):
    img = cv2.imread(image_path)
    
    vis_img = img.copy()
    
    #  K-Means 聚类分离前景和背景
    pixel_values = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixel_values, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    centers = np.uint8(centers)
    bg_cluster_idx = np.argmin(np.mean(centers, axis=1))
    text_cluster_idx = 1 - bg_cluster_idx
    mask = (labels == text_cluster_idx).reshape(img.shape[0], img.shape[1]).astype(np.uint8) * 255
    

    # 图像形态学：上下膨胀，左右限制
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 提取初始基础轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    initial_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 过滤孤立噪点
        if w > 2 and h > 2 and w < img.shape[1] * 0.5:
            initial_boxes.append([x, y, w, h])
            
    if not initial_boxes:
        return []

    # 统计全局字符基准高度中位数
    heights = [b[3] for b in initial_boxes]
    median_h = np.median(heights)

    # 纵向高精度自适应融合策略
    merged_any = True
    while merged_any:
        merged_any = False
        used = [False] * len(initial_boxes)
        
        for i in range(len(initial_boxes)):
            if used[i]: continue
            
            rect1 = initial_boxes[i]
            x1, y1, w1, h1 = rect1
            
            for j in range(i + 1, len(initial_boxes)):
                if used[j]: continue
                
                rect2 = initial_boxes[j]
                x2, y2, w2, h2 = rect2
                
                # 检查横向（X轴）重叠率
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                min_w = min(w1, w2)
                
                if min_w == 0: continue
                overlap_ratio = overlap_x / min_w
                
                if overlap_ratio < 0.5:
                    continue
                
                gap_y = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
                if gap_y > median_h * 0.8: 
                    continue

                if h1 >= median_h * 0.5 and h2 >= median_h * 0.5:
                    continue
                
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                
                initial_boxes[i] = [new_x, new_y, new_w, new_h]
                used[j] = True
                merged_any = True
                
                x1, y1, w1, h1 = initial_boxes[i]
                
        initial_boxes = [initial_boxes[k] for k in range(len(initial_boxes)) if not used[k]]

    # 过滤并绘制最终结果
    final_boxes = []
    for box in initial_boxes:
        x, y, w, h = box
        if h > 10: 
            final_boxes.append(box)

            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    print(f"提取 Box 数量: {len(final_boxes)}")
    
    # 8. 效果可视化
    if flag:
        cv2.imshow("result", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return final_boxes
