def detect_obstacles(frame):
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
