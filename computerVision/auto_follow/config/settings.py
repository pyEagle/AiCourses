# -*- coding:utf-8 -*-

@dataclass
class FollowerConfig:
    model_name  = 'yolov8n.pt'
    camera_id  = 0
    target_class  = 0        # 0: person
    safe_area  = 80000
    area_tolerance  = 10000
    kp_turn  = 0.1
    kp_speed  = 0.0005
    max_output  = 50.0
