import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
import random

# ==================== 0. 全局配置类  ====================
class TrackerConfig:
    def __init__(self):
        # 时间参数 (单位: 秒)，彻底解耦帧率
        self.covered_timeout = 1.0           # 被遮挡多久后进入待删除
        self.hidden_base_timeout = 0.3       # 藏于阴影的基准容忍时间
        self.pending_delete_confirm = 0.5    # 待删除确认期
        self.missing_edge_fast = 0.1         # 边缘向外快速移动丢失容忍时间
        self.missing_edge_slow = 0.5         # 边缘缓慢丢失容忍时间
        self.missing_center = 1.0            # 中心区域无故丢失容忍时间 (防闪烁)
        
        # 空间参数 (比例)
        self.edge_margin_ratio = 0.10        # 定义何为“抽屉边缘”的比例
        
        # 特征画廊配置
        self.gallery_max_size = 5            # 记忆的最大视角数量
        self.gallery_update_thresh = 0.90    # 高于此值仅 EMA 更新，低于此值作为新视角加入
        self.gallery_ema_alpha = 0.1         # EMA 更新率
        
        # 门控阈值
        self.gate_mah_normal = 9.48          # 马氏距离门控
        self.gate_mah_wakeup = 15.0          # 刚唤醒时的马氏距离放宽
        self.gate_cos = 0.5                  # 余弦相似度下限


# ==================== 1. 状态定义 ====================
class ObjectState:
    PLACED = "放入"
    STABLE = "稳定在抽屉中"
    SLOW_MOVING = "碰触/微移"
    FAST_MOVING = "快速拿取/放入"
    SLEEP = "休眠 (抽屉已关闭)"  
    COVERED = "被遮挡/堆叠" 
    HIDDEN_BY_CABINET = "藏于柜内"
    PENDING_DELETE = "待删除确认"
    TAKEN_AWAY = "拿走"


# ==================== 2. 自适应 GMM 模块 ====================
class OnlineSpeedGMM:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.means = np.array([15.0, 150.0, 450.0], dtype=float)
        self.variances = np.array([25.0, 400.0, 2500.0], dtype=float)
        self.weights = np.array([0.33, 0.33, 0.34], dtype=float)
        
        # 注入先验种子，防止全静止冷启动导致三个高斯分布粘合
        self.warmup_buffer = [10.0, 15.0, 140.0, 160.0, 400.0, 500.0] 
        self.warmup_size = 30
        self.is_warmed_up = False

    def update_and_predict(self, speed):
        speed = max(0.0, float(speed))
        if not self.is_warmed_up:
            self.warmup_buffer.append(speed)
            if len(self.warmup_buffer) >= self.warmup_size:
                self._initialize_from_buffer()
                self.is_warmed_up = True
            if speed < 30.0: return 0
            elif speed < 240.0: return 1
            else: return 2

        stds = np.sqrt(self.variances) + 1e-6
        pdfs = self.weights * (1.0 / (np.sqrt(2 * np.pi) * stds)) * np.exp(-0.5 * ((speed - self.means) / stds)**2)
        max_idx, max_pdf = int(np.argmax(pdfs)), np.max(pdfs)
        
        if max_pdf < 1e-3:
            max_idx = int(np.argmin(np.abs(self.means - speed)))

        self.weights = (1 - self.alpha) * self.weights
        self.weights[max_idx] += self.alpha
        self.weights = np.maximum(self.weights, 0.05) 
        self.weights /= np.sum(self.weights)          
        
        old_mean = self.means[max_idx]
        self.means[max_idx] = (1 - self.alpha) * old_mean + self.alpha * speed
        self.variances[max_idx] = max(1.0, (1 - self.alpha) * self.variances[max_idx] + self.alpha * (speed - old_mean)**2)
        self.means[max_idx] = np.clip(self.means[max_idx], [0, 60, 300][max_idx], [45, 240, 900][max_idx])

        if speed < 30.0: return 0  
        return max_idx

    def _initialize_from_buffer(self):
        data = np.array(self.warmup_buffer, dtype=float)
        p33, p66 = np.percentile(data, [33, 66])
        group0, group1, group2 = data[data < p33], data[(data >= p33) & (data < p66)], data[data >= p66]
        if len(group0): self.means[0], self.variances[0] = np.mean(group0), max(1.0, np.var(group0))
        if len(group1): self.means[1], self.variances[1] = np.mean(group1), max(1.0, np.var(group1))
        if len(group2): self.means[2], self.variances[2] = np.mean(group2), max(1.0, np.var(group2))


# ==================== 3. IoU计算与标签平滑 ====================
def compute_iou(boxA, boxB):
    # box: (cx, cy, w, h)
    xA = max(boxA[0] - boxA[2]/2, boxB[0] - boxB[2]/2)
    yA = max(boxA[1] - boxA[3]/2, boxB[1] - boxB[3]/2)
    xB = min(boxA[0] + boxA[2]/2, boxB[0] + boxB[2]/2)
    yB = min(boxA[1] + boxA[3]/2, boxB[1] + boxB[3]/2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

def smooth_label(label_history):
    if not label_history: return None
    scores = {}
    for topk in label_history:
        for label, conf in topk:
            scores[label] = scores.get(label, 0.0) + conf
    return max(scores.items(), key=lambda x: x[1])[0]


# ==================== 4. 特征处理 ====================
class FeatureGallery:
    def __init__(self, initial_emb, config: TrackerConfig):
        self.cfg = config
        norm_emb = initial_emb / (np.linalg.norm(initial_emb) + 1e-6)
        self.gallery = [norm_emb]

    def get_max_sim(self, emb):
        norm_emb = emb / (np.linalg.norm(emb) + 1e-6)
        sims = [float(np.dot(g, norm_emb)) for g in self.gallery]
        return max(sims), np.argmax(sims)

    def update(self, emb, max_sim, best_idx):
        norm_emb = emb / (np.linalg.norm(emb) + 1e-6)
        if max_sim < self.cfg.gallery_update_thresh:
            if len(self.gallery) >= self.cfg.gallery_max_size:
                self.gallery.pop(0)
            self.gallery.append(norm_emb)
        else:
            alpha = self.cfg.gallery_ema_alpha
            self.gallery[best_idx] = (1.0 - alpha) * self.gallery[best_idx] + alpha * norm_emb
            self.gallery[best_idx] /= (np.linalg.norm(self.gallery[best_idx]) + 1e-6)


# ==================== 5. 卡尔曼滤波 ====================
class KinematicKalmanFilter:
    def __init__(self, initial_pos):
        self.kf_dim = 4 # [x, y, vx, vy]
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.P = np.eye(self.kf_dim) * 10.0
        self.R = np.eye(2) * 5.0  
        self._eps = 1e-6

    def predict(self, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        sigma_a_sq = 100.0  
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        Q = sigma_a_sq * np.array([
            [dt4/4, 0,     dt3/2, 0],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0],
            [0,     dt3/2, 0,     dt2]
        ], dtype=float)
        self.x = np.dot(F, self.x)
        self.P = np.dot(np.dot(F, self.P), F.T) + Q
        return self.x[:2].copy()

    def update(self, measurement):
        y = np.array(measurement) - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R + np.eye(2) * self._eps
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.kf_dim)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(K, np.dot(self.R, K.T))
        self.P = np.clip(self.P, 0, 1000.0)

    def get_mahalanobis_distance(self, measurement):
        y = np.array(measurement) - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R + np.eye(2) * self._eps
        return np.sqrt(max(0, np.dot(y.T, np.linalg.solve(S, y))))


# ==================== 6. 时间驱动的 FSM ====================
class FiniteStateMachine:
    def __init__(self, global_gmm, config: TrackerConfig):
        self.cfg = config
        self.state = ObjectState.PLACED
        self.missing_time, self.pending_time = 0.0, 0.0
        self.covered_time, self.hidden_time = 0.0, 0.0
        self.hits = 1
        self.speed_gmm = global_gmm
        self.last_pos = None  
        self.recent_path = deque(maxlen=5) 

    def update(self, matched, velocity, pos, dt, frame_info, is_covered=False):
        if not matched:
            if is_covered:
                self.state = ObjectState.COVERED
                self.covered_time += dt
                self.missing_time = 0.0
                if self.covered_time > self.cfg.covered_timeout:
                    self.state = ObjectState.PENDING_DELETE
                    self.pending_time = 0.0
                return

            self.missing_time += dt
            x, y = self.last_pos if self.last_pos is not None else pos
            d_xmin, d_ymin, d_xmax, d_ymax = frame_info.get("drawer_bbox", [0, 0, 640, 480])
            d_w, d_h = max(1.0, d_xmax - d_xmin), max(1.0, d_ymax - d_ymin)
            
            if (x < d_xmin or x > d_xmax or y < d_ymin or y > d_ymax):
                self.state = ObjectState.HIDDEN_BY_CABINET
                self.hidden_time += dt
                self.missing_time = 0.0
                open_ratio = float(np.clip(frame_info.get("drawer_open_ratio", 1.0), 0.05, 1.0))
                timeout = self.cfg.hidden_base_timeout + (1.0 - open_ratio) * 1.5 
                if self.hidden_time > timeout:
                    self.state, self.pending_time = ObjectState.PENDING_DELETE, 0.0
                return

            margin_x, margin_y = d_w * self.cfg.edge_margin_ratio, d_h * self.cfg.edge_margin_ratio
            is_at_edge = (x < d_xmin + margin_x or x > d_xmax - margin_x or 
                          y < d_ymin + margin_y or y > d_ymax - margin_y)
            
            speed = np.linalg.norm(velocity)
            tol_time = self.cfg.missing_center # 默认中心区域容忍
            if is_at_edge and speed > 1e-3:
                nx = -1 if x < d_xmin + margin_x else (1 if x > d_xmax - margin_x else 0)
                ny = -1 if y < d_ymin + margin_y else (1 if y > d_ymax - margin_y else 0)
                norm_n = np.linalg.norm([nx, ny])
                if norm_n > 0:
                    dot_product = (velocity[0] * (nx/norm_n) + velocity[1] * (ny/norm_n)) / speed
                    tol_time = self.cfg.missing_edge_fast if dot_product > 0.3 else self.cfg.missing_edge_slow
            
            if self.missing_time > tol_time and self.state != ObjectState.PENDING_DELETE:
                self.state, self.pending_time = ObjectState.PENDING_DELETE, 0.0
            
            if self.state == ObjectState.PENDING_DELETE:
                self.pending_time += dt
                if self.pending_time > self.cfg.pending_delete_confirm:
                    self.state = ObjectState.TAKEN_AWAY
            return

        self.covered_time, self.hidden_time = 0.0, 0.0
        if self.state == ObjectState.PENDING_DELETE: self.state = ObjectState.STABLE
            
        self.last_pos = pos 
        self.recent_path.append(pos)
        self.missing_time, self.pending_time = 0.0, 0.0
        self.hits += 1
        
        speed_rank = self.speed_gmm.update_and_predict(np.linalg.norm(velocity))
        if self.hits < 3: self.state = ObjectState.PLACED
        elif speed_rank == 0: self.state = ObjectState.STABLE       
        elif speed_rank == 1: self.state = ObjectState.SLOW_MOVING  
        else: self.state = ObjectState.FAST_MOVING  


# ==================== 7. Track 逻辑 ====================
class Track:
    def __init__(self, target_id, embedding, pos, size, label_topk3, global_gmm, config: TrackerConfig):
        self.track_id = target_id
        self.cfg = config
        self.kf = KinematicKalmanFilter(pos)
        self.fsm = FiniteStateMachine(global_gmm, config)
        self.gallery = FeatureGallery(embedding, config)
        
        self.label_history = deque([label_topk3], maxlen=9)
        self.current_pos = np.array(pos, dtype=float)
        self.size = np.array(size, dtype=float) # [w, h]

    def update(self, embedding, pos, size, max_sim, best_idx, dt, frame_info):
        self.current_pos = np.array(pos, dtype=float)

        self.size = 0.8 * self.size + 0.2 * np.array(size, dtype=float)
        
        self.gallery.update(embedding, max_sim, best_idx)
        
        velocity_pred = self.kf.x[2:4].copy()
        self.fsm.update(matched=True, velocity=velocity_pred, pos=pos, dt=dt, frame_info=frame_info)
        self.kf.update(self.current_pos)
        self.label_history.append(label_topk3)

    def miss(self, dt, frame_info, is_covered=False):
        pred_pos = self.kf.x[:2].copy()
        self.current_pos = pred_pos
        self.fsm.update(matched=False, velocity=self.kf.x[2:4], pos=pred_pos, dt=dt, frame_info=frame_info, is_covered=is_covered)


# ==================== 8. IoU融合关联 ====================
class MultiObjectTracker:
    def __init__(self, config=None):
        self.cfg = config if config else TrackerConfig()
        self.tracks = []
        self.next_id = 0
        self.global_speed_gmm = OnlineSpeedGMM(alpha=0.15)
        self._last_raw_open, self._debounce_count = True, 0
        self.stable_drawer_open, self.is_sleeping = True, False 

    def step(self, cur_frame, frame_info=None, dt=0.033):
        # cur_frame 格式约定: [[embedding, (cx, cy), [labels], (w, h)], ...]
        if frame_info is None:
            frame_info = {"drawer_open": True, "drawer_bbox": [0,0,640,480], "drawer_open_ratio": 1.0}

        raw_open = frame_info.get("drawer_open", True)
        if raw_open != self._last_raw_open: self._debounce_count = 0
        self._debounce_count += 1
        self._last_raw_open = raw_open
        if self._debounce_count >= 5: self.stable_drawer_open = raw_open

        if not self.stable_drawer_open:
            self.is_sleeping = True 
            for track in self.tracks:
                track.fsm.state = ObjectState.SLEEP
            return self._build_results()

        just_woke_up = False
        if self.is_sleeping and self.stable_drawer_open:
            just_woke_up, self.is_sleeping = True, False
            for track in self.tracks: track.kf.P += np.eye(track.kf.kf_dim) * 100.0  

        for track in self.tracks: 
            if track.fsm.state in [ObjectState.COVERED, ObjectState.HIDDEN_BY_CABINET, ObjectState.PENDING_DELETE]:
                track.kf.x[2:4] = 0.0 
                track.kf.P += np.eye(track.kf.kf_dim) * 10.0 
                track.kf.P = np.clip(track.kf.P, 0, 500.0) 
            track.kf.predict(dt)
            
        num_tracks, num_dets = len(self.tracks), len(cur_frame)
        gate_mah = self.cfg.gate_mah_wakeup if just_woke_up else self.cfg.gate_mah_normal  
        
        matched_tracks, matched_dets = set(), set()
        if num_tracks > 0 and num_dets > 0:
            cost_matrix = np.full((num_tracks, num_dets), 1e5)
            meta_data = {} # 暂存 max_sim 和 best_idx 避免重复计算
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(cur_frame):
                    det_emb, det_pos = det[0], det[1]
                    d_m = track.kf.get_mahalanobis_distance(det_pos)
                    max_sim, best_idx = track.gallery.get_max_sim(det_emb)
                    
                    if d_m > gate_mah or max_sim < self.cfg.gate_cos: continue 
                    w_spa = 0.5 * np.exp(-d_m / 10.0) if just_woke_up else 0.7 * np.exp(-d_m / 3.0)
                    cost_matrix[i, j] = (1.0 - w_spa) * (1.0 - max_sim) + w_spa * (d_m / gate_mah)
                    meta_data[(i, j)] = (max_sim, best_idx)
                    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < (1.2 if just_woke_up else 0.8):
                    matched_tracks.add(r)
                    matched_dets.add(c)
                    max_sim, best_idx = meta_data[(r, c)]
                    self.tracks[r].update(cur_frame[c][0], cur_frame[c][1], cur_frame[c][3], max_sim, best_idx, dt, frame_info) 

        for i, track in enumerate(self.tracks):
            if i not in matched_tracks: 
                is_covered = False
                pred_box = (track.current_pos[0], track.current_pos[1], track.size[0], track.size[1])
                for j, det in enumerate(cur_frame):
                    det_box = (det[1][0], det[1][1], det[3][0], det[3][1])
                    if compute_iou(pred_box, det_box) > 0.3: # IoU > 0.3 认为被遮挡
                        is_covered = True
                        break
                track.miss(dt, frame_info=frame_info, is_covered=is_covered)

        for j, det in enumerate(cur_frame):
            if j not in matched_dets:
                self.tracks.append(Track(self.next_id, det[0], det[1], det[3], det[2], self.global_speed_gmm, self.cfg))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.fsm.state != ObjectState.TAKEN_AWAY]
        return self._build_results()

    def _build_results(self):
        return [{
            "track_id": t.track_id, "position": np.round(t.current_pos, 1), "size": np.round(t.size, 1),
            "label": smooth_label(list(t.label_history)), "fsm_state": t.fsm.state
        } for t in self.tracks]


if __name__ == "__main__":
    frame_info_open = {"drawer_open": True, "drawer_bbox": [50, 50, 590, 430], "drawer_open_ratio": 1.0}
    dt_val = 0.033 # 模拟 30FPS，全系统已解耦为真实时间

    print("\n========= ReID + IoU 追踪器验证 =========")
    
    # 测试 1: 翻面 
    print("\n--- 测试 1: 物品原地翻面  测试 ---")
    tracker = MultiObjectTracker()
    emb_front = np.array([1.0, 0.0, 0.0])
    emb_back = np.array([0.0, 1.0, 0.0]) # 翻面后，外观特征完全改变 (cos_sim = 0.0)
    
    # 放入正面
    cur_frame_front = [[emb_front, (300.0, 200.0), [("说明书", 0.9)], (50.0, 50.0)]]
    tracker.step(cur_frame_front, frame_info_open, dt=dt_val)
    print(f"  放入时 Track ID: {tracker.tracks[0].track_id}, 容量: {len(tracker.tracks[0].gallery.gallery)}")
    
    # 物品在极短时间内原地翻转
    cur_frame_back = [[emb_back, (301.0, 201.0), [("说明书", 0.9)], (50.0, 50.0)]]
    tracker.step(cur_frame_back, frame_info_open, dt=dt_val)
    print(f"  翻面后 Track ID (应不变): {tracker.tracks[0].track_id}, 画廊容量(应增至2): {len(tracker.tracks[0].gallery.gallery)}")
    
    # 测试 2: 基于 IoU 的严谨遮挡测试
    print("\n--- 测试 2: 基于 IoU 交并比的精准遮挡测试 ---")
    tracker = MultiObjectTracker()
    # 两个物品并排放置，中心点较近 (300 vs 320)，但宽高 (20) 使它们没有重合，IoU=0
    cur_frame_parallel = [
        [np.array([1.0, 0.0, 0.0]), (300.0, 200.0), [("A药", 0.9)], (20.0, 20.0)],
        [np.array([0.0, 1.0, 0.0]), (330.0, 200.0), [("B药", 0.9)], (20.0, 20.0)]
    ]
    tracker.step(cur_frame_parallel, frame_info_open, dt=dt_val)
    print(f"  初始并排放置: 存活 Track 数量: {len(tracker.tracks)}")
    
    # 模拟 B药漏检，但 A药仍在原处
    cur_frame_miss_B = [[np.array([1.0, 0.0, 0.0]), (300.0, 200.0), [("A药", 0.9)], (20.0, 20.0)]]
    tracker.step(cur_frame_miss_B, frame_info_open, dt=dt_val)
    # 因为 IoU 为 0，B 药不会被误判为被遮挡，而是正常的缺失 (PENDING_DELETE 倒计时)
    track_b = next((t for t in tracker.tracks if t.track_id == 1), None)
    print(f"  并排漏检: B药由于没有真正重叠，状态判定为: {track_b.fsm.state} (而非 COVERED)")
    
    # 如果A药是一个超大盒子(100x100)，完全盖住了B药的预测位置，就会触发 IoU 遮挡
    cur_frame_huge_A = [[np.array([1.0, 0.0, 0.0]), (315.0, 200.0), [("A药大盒", 0.9)], (100.0, 100.0)]]
    tracker.step(cur_frame_huge_A, frame_info_open, dt=dt_val)
    print(f"  大盒盖上: B药与大盒产生极高IoU，状态判定为: {track_b.fsm.state}")

    print("\n--- 全部测试完成 ---")
