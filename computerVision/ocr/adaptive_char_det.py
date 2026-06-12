# -*- coding:utf-8 -*-

import cv2
import numpy as np

class AdaptiveCharacterDetector:
    def __init__(self):
        self.img = None
        self.vis = None
        self.mask = None
        self.median_w = 0
        self.median_h = 0
        self.median_area = 0

    def _generate_mask(self):
        """K-Means 聚类生成掩码"""
        pixels = self.img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
        _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        centers = np.uint8(centers)
        brightness = np.mean(centers, axis=1)
        text_cluster = np.argmax(brightness)
        
        h_img, w_img = self.img.shape[:2]
        self.mask = (labels.reshape(h_img, w_img) == text_cluster).astype(np.uint8) * 255

    def _compute_stats(self):
        """计算字符基准统计信息"""
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(self.mask, connectivity=8)
        widths, heights, areas = [], [], []
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 2:
                widths.append(stats[i, cv2.CC_STAT_WIDTH])
                heights.append(stats[i, cv2.CC_STAT_HEIGHT])
                areas.append(stats[i, cv2.CC_STAT_AREA])
        
        if not areas: return False
        self.median_w, self.median_h, self.median_area = np.median(widths), np.median(heights), np.median(areas)
        
        # 闭运算修正
        kernel_h = min(3, max(1, int(self.median_h * 0.03)))
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h)))
        return True

    def _split_rows(self):
        """行分割"""
        projection = np.sum(self.mask > 0, axis=1)
        smooth_kernel = max(3, self.mask.shape[0] // 200)
        projection = cv2.blur(projection.astype(np.float32).reshape(-1, 1), (1, smooth_kernel)).flatten()
        threshold = np.max(projection) * 0.05
        
        rows, in_text, start_y = [], False, 0
        for y, v in enumerate(projection):
            if v > threshold and not in_text:
                start_y, in_text = y, True
            elif v <= threshold and in_text:
                if y - start_y > 5: rows.append((start_y, y))
                in_text = False
        if in_text: rows.append((start_y, len(projection) - 1))
        return rows

    def _merge_boxes(self, boxes):
        """垂直方向碎片合并"""
        merged = True
        while merged:
            merged = False
            used = [False] * len(boxes)
            new_boxes = []
            for i in range(len(boxes)):
                if used[i]: continue
                x1, y1, w1, h1 = boxes[i]
                for j in range(i + 1, len(boxes)):
                    if used[j]: continue
                    x2, y2, w2, h2 = boxes[j]
                    
                    overlap_ratio = max(0, min(x1+w1, x2+w2) - max(x1, x2)) / max(1, min(w1, w2))
                    gap_y = max(0, max(y1, y2) - min(y1+h1, y2+h2))
                    
                    if overlap_ratio >= 0.6 and gap_y <= self.median_h * 0.5:
                        x1, y1, w1, h1 = min(x1, x2), min(y1, y2), max(x1+w1, x2+w2)-min(x1, x2), max(y1+h1, y2+h2)-min(y1, y2)
                        used[j] = True
                        merged = True
                new_boxes.append([x1, y1, w1, h1])
            boxes = new_boxes
        return boxes

    def run(self, image_path, flag=True):
        self.img = cv2.imread(image_path)
        
        self.vis = self.img.copy()
        self._generate_mask()
        if not self._compute_stats():
            print("未检测到有效字符")
            return []

        rows = self._split_rows()
        final_boxes = []

        for y1, y2 in rows:
            row_mask = self.mask[y1:y2, :]
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(row_mask, connectivity=8)
            boxes = [[stats[i, 0], stats[i, 1]+y1, stats[i, 2], stats[i, 3]] 
                     for i in range(1, num_labels) 
                     if stats[i, cv2.CC_STAT_AREA] >= self.median_area * 0.03 and stats[i, cv2.CC_STAT_HEIGHT] >= self.median_h * 0.15]
            
            boxes = self._merge_boxes(boxes)
            final_boxes.extend(sorted(boxes, key=lambda b: b[0]))

        # 可视化
        if flag:
            for idx, (x, y, w, h) in enumerate(sorted(final_boxes, key=lambda b: (b[1], b[0]))):
                cv2.rectangle(self.vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Result", self.vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return final_boxes

if __name__ == "__main__":
    import sys
    detector = AdaptiveCharacterDetector()
    detector.run(sys.argv[1])
  
