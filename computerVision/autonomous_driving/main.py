# -*- coding:utf-8 -*-

import cv2
import numpy as np

from ultralytics import YOLO

class AutonomousDrivingSystem:
    def __init__(self, video_path):
        self.video_path = video_path
        print("正在加载 YOLOv8 模型...")
        self.yolo_model = YOLO("yolov8n.pt") 
        
    def detect_lanes(self, frame):
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

    def detect_obstacles(self, frame):
        results = self.yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        obstacles = []
        for box in results[0].boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            # 只关注车辆 (2, 3, 5, 7) 和行人 (0)
            if cls in [0, 2, 3, 5, 7]:
                center_x = (x1 + x2) / 2
                bottom_y = y2 # 取底部 Y 坐标代表距离
                obstacles.append((center_x, bottom_y))
                
        return annotated_frame, obstacles

    def reconstruct_2d_map(self, frame_shape, obstacles):
        h, w = frame_shape[:2]
        map_size = 400
        bev_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        
        # 绘制本车 (地图底部中央)
        ego_x, ego_y = map_size // 2, map_size - 50
        cv2.rectangle(bev_map, (ego_x - 10, ego_y - 20), (ego_x + 10, ego_y + 20), (0, 255, 0), -1)
        
        danger_zone = False
        
        # 将前视障碍物映射到 BEV (透视投影的简化表达)
        for (obs_x, obs_y) in obstacles:
            # 归一化位置
            norm_x = obs_x / w
            norm_y = (h - obs_y) / h # 越靠下，距离越近
            
            # 映射到 2D 地图
            map_x = int(norm_x * map_size)
            map_y = int(ego_y - (1 - norm_y) * (map_size - 100))
            
            cv2.circle(bev_map, (map_x, map_y), 10, (0, 0, 255), -1)
            
            # 碰撞检测：如果障碍物在正前方且距离很近
            if abs(map_x - ego_x) < 40 and map_y > ego_y - 100:
                danger_zone = True
                
        return bev_map, danger_zone

    def navigate(self, center_offset, danger_zone):
        if danger_zone:
            return "BRAKE (Obstacle Ahead!)", (0, 0, 255)
        elif center_offset == -1:
            return "STEER RIGHT (Adjust Lane)", (0, 255, 255)
        elif center_offset == 1:
            return "STEER LEFT (Adjust Lane)", (0, 255, 255)
        else:
            return "CRUISE (Path Clear)", (0, 255, 0)

    def run(self):
        cap = cv2.VideoCapture(self.video_path) 
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, (800, 480))
            
            # 1. 车道线检测
            lane_frame, center_offset = self.detect_lanes(frame.copy())
            
            # 2. 障碍物检测
            yolo_frame, obstacles = self.detect_obstacles(frame.copy())
            
            # 3. 2D 局部建图 (雷达图)
            bev_map, danger_zone = self.reconstruct_2d_map(frame.shape, obstacles)
            
            # 4. 决策规划
            action_text, text_color = self.navigate(center_offset, danger_zone)
            
            # 融合车道线和 YOLO 结果
            main_view = cv2.addWeighted(lane_frame, 0.5, yolo_frame, 0.5, 0)
            cv2.putText(main_view, f"Status: {action_text}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
            
            # 将 BEV 贴在主视图右下角
            bev_resized = cv2.resize(bev_map, (200, 200))
            main_view[280:480, 600:800] = bev_resized
            cv2.rectangle(main_view, (600, 280), (800, 480), (255, 255, 255), 2)
            cv2.putText(main_view, "Local BEV Map", (610, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Autonomous Driving UI", main_view)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 使用 0 表示使用本地摄像头，或者传入一段行车记录仪 MP4 视频的路径
    system = AutonomousDrivingSystem(0) 
    system.run()
  
