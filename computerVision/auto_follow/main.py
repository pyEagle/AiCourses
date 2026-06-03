import cv2
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from ultralytics import YOLO


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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class VisionFollower:
    def __init__(self, config):
        self.cfg = config
        self.model = self._init_model()
        self.cap = self._init_camera()
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2

    def _init_model(self):
        logger.info(f"Loading YOLO model: {self.cfg.model_name}...")
        try:
            return YOLO(self.cfg.model_name)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _init_camera(self):
        logger.info(f"Opening camera {self.cfg.camera_id}...")
        cap = cv2.VideoCapture(self.cfg.camera_id)
        if not cap.isOpened():
            logger.error(f"Cannot open camera ID {self.cfg.camera_id}")
            raise RuntimeError("Camera initialization failed.")
        return cap

    def calculate_control_signals(self, largest_area, target_cx):
        error_x = target_cx - self.center_x
        motor_turn = error_x * self.cfg.kp_turn
        
        error_area = self.cfg.safe_area - largest_area
        
        if abs(error_area) < self.cfg.area_tolerance:
            motor_speed = 0.0
            action_text = "Status: HOLD POSITION"
        else:
            motor_speed = error_area * self.cfg.kp_speed
            action_text = "Status: MOVING FORWARD" if motor_speed > 0 else "Status: MOVING BACKWARD"


        motor_turn = max(min(motor_turn, self.cfg.max_output), -self.cfg.max_output)
        motor_speed = max(min(motor_speed, self.cfg.max_output), -self.cfg.max_output)
        
        return motor_turn, motor_speed, action_text

    def run(self):
        logger.info("System initialized. Starting auto-follow loop. Press 'q' to quit.")
        
        try:
            while True:
                start_time = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to grab frame. Exiting loop.")
                    break

                frame = cv2.flip(frame, 1)
                
                results = self.model.track(frame, classes=[self.cfg.target_class], persist=True, verbose=False)
                
                motor_turn, motor_speed = 0.0, 0.0
                action_text = "Status: SEARCHING"

                if results and results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    
                    largest_area = 0
                    best_box = None
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        area = (x2 - x1) * (y2 - y1)
                        if area > largest_area:
                            largest_area = area
                            best_box = box

                    if best_box is not None:
                        x1, y1, x2, y2 = map(int, best_box[:4])
                        target_cx, target_cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (target_cx, target_cy), 5, (0, 0, 255), -1)
                        cv2.line(frame, (self.center_x, self.center_y), (target_cx, target_cy), (255, 0, 0), 2)

                        motor_turn, motor_speed, action_text = self.calculate_control_signals(largest_area, target_cx)


                fps = 1.0 / (time.time() - start_time + 1e-5) # 防止除零异常
                cv2.line(frame, (self.center_x - 20, self.center_y), (self.center_x + 20, self.center_y), (0, 255, 255), 2)
                cv2.line(frame, (self.center_x, self.center_y - 20), (self.center_x, self.center_y + 20), (0, 255, 255), 2)
                
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, action_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                turn_dir = "RIGHT" if motor_turn > 0 else "LEFT" if motor_turn < 0 else "CENTER"
                cv2.putText(frame, f"Motor Turn: {motor_turn:.1f} [{turn_dir}]", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
                cv2.putText(frame, f"Motor Speed: {motor_speed:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

                cv2.imshow("Vision Based Auto-Following", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested shutdown.")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Process interrupted by user (Ctrl+C).")
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Releasing resources...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("System shutdown complete.")

def main():
    config = FollowerConfig()
    follower = VisionFollower(config)
    follower.run()

if __name__ == "__main__":
    main()

