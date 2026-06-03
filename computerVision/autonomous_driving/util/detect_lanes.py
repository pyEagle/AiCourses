import cv2
import numpy as np

def detect_lanes(frame):
      height, width = frame.shape[:2]
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      blur = cv2.GaussianBlur(gray, (5, 5), 0)
      edges = cv2.Canny(blur, 50, 150)
      
      # 定义感兴趣区域 (ROI) - 假设摄像头安装在车头正前
      polygons = np.array([
          [(0, height), (width, height), (width, int(height*0.6)), (0, int(height*0.6))]
      ])
      mask = np.zeros_like(edges)
      cv2.fillPoly(mask, polygons, 255)
      masked_edges = cv2.bitwise_and(edges, mask)
      
      # 霍夫变换检测直线
      lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=20)
      
      lane_image = np.zeros_like(frame)
      center_offset = 0 # 车道中心偏移量
      
      if lines is not None:
          left_lines = []
          right_lines = []
          for line in lines:
              x1, y1, x2, y2 = line[0]
              parameters = np.polyfit((x1, x2), (y1, y2), 1)
              slope = parameters[0]
              if slope < -0.5: # 左侧车道线
                  left_lines.append(line)
              elif slope > 0.5: # 右侧车道线
                  right_lines.append(line)
                  
          # 绘制所有线段
          for line in lines:
              x1, y1, x2, y2 = line[0]
              cv2.line(lane_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
              
          # 估算偏移量 (简易逻辑：比较左右线段数量或位置)
          # 简化为：如果只有一边线，则向另一边微调
          if len(left_lines) > len(right_lines):
              center_offset = 1 # 偏右
          elif len(right_lines) > len(left_lines):
              center_offset = -1 # 偏左

      return cv2.addWeighted(frame, 0.8, lane_image, 1, 0), center_offset
