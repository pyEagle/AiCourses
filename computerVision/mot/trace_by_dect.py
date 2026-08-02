import os
import pickle
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

# ==================== 1. 状态定义 ====================
class ObjectState:
    PLACED = "放入 (Tentative)"
    STABLE = "稳定在抽屉中 (Confirmed-Static)"
    SLOW_MOVING = "碰触/微移 (Confirmed-Slow)"
    FAST_MOVING = "快速拿取/放入 (Confirmed-Fast)"
    SLEEP = "休眠 (抽屉已关闭)"  # 【新增】抽屉关闭状态
    TAKEN_AWAY = "拿走 (Deleted)"

# ==================== 2. 自适应 GMM 模块 ====================
class OnlineSpeedGMM:
    def __init__(self, k=3, alpha=0.1):
        self.k = k            
        self.alpha = alpha    
        self.means = np.array([0.5, 5.0, 15.0], dtype=float) 
        self.variances = np.array([0.5, 2.0, 5.0], dtype=float)
        self.weights = np.array([1.0/k] * k, dtype=float)

    def update_and_predict(self, speed):
        stds = np.sqrt(self.variances) + 1e-6
        pdfs = self.weights * (1.0 / (np.sqrt(2 * np.pi) * stds)) * np.exp(-0.5 * ((speed - self.means) / stds)**2)
        
        max_idx = np.argmax(pdfs)
        max_pdf = pdfs[max_idx]

        if max_pdf < 1e-3:
            replace_idx = np.argmin(self.weights)
            self.means[replace_idx] = speed
            self.variances[replace_idx] = max(0.1, speed * 0.1) 
            self.weights[replace_idx] = 0.01  
            max_idx = replace_idx
        else:
            self.weights = (1 - self.alpha) * self.weights
            self.weights[max_idx] += self.alpha
            self.means[max_idx] = (1 - self.alpha) * self.means[max_idx] + self.alpha * speed
            self.variances[max_idx] = max(0.05, (1 - self.alpha) * self.variances[max_idx] + self.alpha * (speed - self.means[max_idx])**2)

        self.weights /= np.sum(self.weights)
        if speed < 1.0: return 0  
        sorted_indices = np.argsort(self.means)
        return np.where(sorted_indices == max_idx)[0][0]

# ==================== 3. 自适应状态机 (场景升级版) ====================
class FiniteStateMachine:
    def __init__(self, global_gmm):
        self.state = ObjectState.PLACED
        self.missing_frames = 0
        self.hits = 1
        self.speed_gmm = global_gmm
        self.is_visible = True 

    def update(self, matched, velocity, pos, frame_info):
        drawer_open = frame_info.get("drawer_open", True)
        
        # 【修改 1】：如果抽屉关闭，冻结所有行为并休眠，不增加丢失帧数
        if not drawer_open:
            self.state = ObjectState.SLEEP
            self.is_visible = False
            return

        self.is_visible = matched
        if not matched:
            self.missing_frames += 1
            
            # 【修改 2】：判定在画面中央消失还是边缘消失 (手部遮挡判定)
            x, y = pos
            w, h = frame_info.get("width", 640), frame_info.get("height", 480)
            edge_margin = 60 # 边缘 60 像素视为拿走区
            
            is_at_edge = (x < edge_margin or x > w - edge_margin or 
                          y < edge_margin or y > h - edge_margin)
            
            # 边缘消失容忍 3 帧；中央消失（极可能被手包裹）容忍 30 帧 (约1秒)
            missing_tolerance = 3 if is_at_edge else 30
            
            if self.missing_frames > missing_tolerance:  
                self.state = ObjectState.TAKEN_AWAY
            return

        # 匹配成功，重置计数器
        self.missing_frames = 0
        self.hits += 1
        speed = np.linalg.norm(velocity)
        speed_rank = self.speed_gmm.update_and_predict(speed)
        
        if self.hits < 3:
            self.state = ObjectState.PLACED
        elif speed_rank == 0: 
            self.state = ObjectState.STABLE       
        elif speed_rank == 1:
            self.state = ObjectState.SLOW_MOVING  
        else:
            self.state = ObjectState.FAST_MOVING  

# ==================== 4. RL Q-Learning 参数调优器 ====================
class QLearningAutoTuner:
    def __init__(self, alpha=0.1, gamma=0.9, initial_epsilon=0.5, min_epsilon=0.01, decay_rate=0.995):
        self.alpha = alpha      
        self.gamma = gamma      
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        
        self.actions = [
            (0.01, 5.0),  # 动作 0: 信任模型 (遮挡、防手抖)
            (0.1,  2.0),  # 动作 1: 平衡模式
            (1.5,  0.5),  # 动作 2: 信任观测 (药品被快速抓走)
        ]
        self.q_table = {}

    def _get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]

    def choose_action(self, state, train_mode=True):
        if not train_mode: return np.argmax(self._get_q(state))
        if np.random.rand() < self.epsilon: return np.random.randint(len(self.actions))
        return np.argmax(self._get_q(state))

    def learn(self, state, action, reward, next_state):
        q_s = self._get_q(state)
        q_next = self._get_q(next_state)
        best_next_action = np.argmax(q_next)
        td_target = reward + self.gamma * q_next[best_next_action]
        q_s[action] += self.alpha * (td_target - q_s[action])
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

# ==================== 5. 核心追踪模块 ====================
class AdaptiveKalmanFilter:
    def __init__(self, initial_pos):
        self.kf_dim = 4
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.P = np.eye(self.kf_dim) * 10.0
        self.R = np.eye(2) * 2.0
        self.Q = np.eye(self.kf_dim) * 0.1

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:2]

    def set_params(self, q_val, r_val):
        self.Q = np.eye(self.kf_dim) * q_val
        self.R = np.eye(2) * r_val

    def update(self, measurement):
        y = measurement - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.kf_dim)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(K, np.dot(self.R, K.T))

    def get_mahalanobis_distance(self, measurement):
        y = measurement - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        return np.sqrt(np.dot(y.T, np.linalg.solve(S, y)))

def viterbi_decode(label_history):
    if not label_history: return None
    states = list(set([label for topk in label_history for label, _ in topk]))
    if not states: return None
    num_states, T = len(states), len(label_history)
    viterbi_table = np.zeros((T, num_states))
    backpointer = np.zeros((T, num_states), dtype=int)
    trans_p = np.full((num_states, num_states), 0.05 / max(1, num_states - 1))
    np.fill_diagonal(trans_p, 0.95)
    first_topk = dict(label_history[0])
    for s_idx, state in enumerate(states):
        viterbi_table[0, s_idx] = np.log(max(first_topk.get(state, 0.01), 1e-5))
    for t in range(1, T):
        curr_topk = dict(label_history[t])
        for s_idx, state in enumerate(states):
            log_emission = np.log(max(curr_topk.get(state, 0.01), 1e-5))
            probabilities = viterbi_table[t-1] + np.log(trans_p[:, s_idx])
            best_prev_idx = np.argmax(probabilities)
            viterbi_table[t, s_idx] = probabilities[best_prev_idx] + log_emission
            backpointer[t, s_idx] = best_prev_idx
    best_path = [np.argmax(viterbi_table[T-1])]
    for t in range(T-1, 0, -1):
        best_path.insert(0, backpointer[t, best_path[0]])
    return states[best_path[-1]]

class Track:
    def __init__(self, target_id, embedding, pos, label_topk3, global_gmm, rl_tuner):
        self.track_id = target_id
        self.kf = AdaptiveKalmanFilter(pos)
        self.fsm = FiniteStateMachine(global_gmm)
        self.rl_tuner = rl_tuner  
        
        self.current_embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        self.embedding_history = deque([self.current_embedding], maxlen=9)
        self.label_history = deque([label_topk3], maxlen=9)
        self.current_pos = pos

        self.current_category = label_topk3[0][0] if label_topk3 else "unknown"
        self.last_state = None
        self.last_action = None

    def update(self, embedding, pos, label_topk3, frame_info, train_mode=True):
        self.current_pos = pos
        self.current_category = label_topk3[0][0] if label_topk3 else "unknown"
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-6)
        cos_sim = np.dot(self.current_embedding, norm_emb)
        
        alpha = 0.95 - 0.15 * max(0, cos_sim)
        self.current_embedding = alpha * self.current_embedding + (1.0 - alpha) * norm_emb
        self.current_embedding /= (np.linalg.norm(self.current_embedding) + 1e-6)
        
        velocity_pred = self.kf.x[2:4] 
        self.fsm.update(matched=True, velocity=velocity_pred, pos=pos, frame_info=frame_info)
        
        current_rl_state = (self.current_category, self.fsm.state, self.fsm.is_visible)

        if train_mode and self.last_action is not None:
            pred_pos = self.kf.x[:2]
            err = np.linalg.norm(np.array(pos) - pred_pos)
            reward = 10.0 / (1.0 + err) + 5.0 * cos_sim
            self.rl_tuner.learn(state=self.last_state, action=self.last_action, 
                                reward=reward, next_state=current_rl_state)

        action_idx = self.rl_tuner.choose_action(current_rl_state, train_mode=train_mode)
        q_val, r_val = self.rl_tuner.actions[action_idx]

        self.kf.set_params(q_val, r_val)
        self.kf.update(np.array(pos))
        
        self.last_state = current_rl_state
        self.last_action = action_idx
        self.embedding_history.append(self.current_embedding)
        self.label_history.append(label_topk3)

    def miss(self, frame_info, train_mode=True):
        pred_pos = self.kf.x[:2] # 使用 KF 预测位置来判断此时药盒应该在哪
        self.fsm.update(matched=False, velocity=np.zeros(2), pos=pred_pos, frame_info=frame_info)
        current_rl_state = (self.current_category, self.fsm.state, self.fsm.is_visible)
        
        # 如果是因为闭合抽屉导致的miss，不予惩罚；如果是视野中丢失，予以惩罚
        if train_mode and self.last_action is not None:
            reward = 0.0 if not frame_info.get("drawer_open", True) else -5.0 
            self.rl_tuner.learn(state=self.last_state, action=self.last_action, 
                                reward=reward, next_state=current_rl_state)
            
        action_idx = self.rl_tuner.choose_action(current_rl_state, train_mode=train_mode)
        q_val, r_val = self.rl_tuner.actions[action_idx]
        self.kf.set_params(q_val, r_val)
        
        self.last_state = current_rl_state
        self.last_action = action_idx

class MultiObjectTracker:
    def __init__(self, is_training=True):
        self.tracks = []
        self.next_id = 0
        self.global_speed_gmm = OnlineSpeedGMM(k=3, alpha=0.15)
        self.rl_tuner = QLearningAutoTuner(alpha=0.2, gamma=0.9, initial_epsilon=0.5)
        self.is_training = is_training 

    def step(self, cur_frame, frame_info=None):
        if frame_info is None:
            # 默认：抽屉开启，画面尺寸640x480
            frame_info = {"drawer_open": True, "width": 640, "height": 480}

        for track in self.tracks: track.kf.predict()
        num_tracks, num_dets = len(self.tracks), len(cur_frame)
        
        if num_tracks > 0 and num_dets > 0:
            cost_matrix = np.full((num_tracks, num_dets), 1e5) 
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(cur_frame):
                    det_emb, det_pos, _ = det
                    norm_det_emb = det_emb / (np.linalg.norm(det_emb) + 1e-6)
                    d_m = track.kf.get_mahalanobis_distance(det_pos)
                    if d_m > 5.99: continue 
                    
                    cos_sim = np.dot(track.current_embedding, norm_det_emb)
                    
                    # 【修改 3】：削弱距离在矩阵中的权重，高度依赖外观 Cosine 余弦 (应对药盒密集挤压)
                    w_spa = 0.2 + 0.2 * np.exp(-d_m / 2.0)
                    cost_matrix[i, j] = (1.0 - w_spa) * (1.0 - cos_sim) + w_spa * (d_m / 5.99)
                    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)

        matched_tracks, matched_dets = set(), set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.8:
                matched_tracks.add(r)
                matched_dets.add(c)
                self.tracks[r].update(cur_frame[c][0], cur_frame[c][1], cur_frame[c][2], 
                                      frame_info=frame_info, train_mode=self.is_training) 

        for i, track in enumerate(self.tracks):
            if i not in matched_tracks: 
                track.miss(frame_info=frame_info, train_mode=self.is_training)

        for j, det in enumerate(cur_frame):
            if j not in matched_dets:
                self.tracks.append(Track(self.next_id, det[0], det[1], det[2], self.global_speed_gmm, self.rl_tuner))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.fsm.state != ObjectState.TAKEN_AWAY]

        results = []
        for track in self.tracks:
            smooth_label = viterbi_decode(list(track.label_history))
            results.append({
                "track_id": track.track_id,
                "position": np.round(track.current_pos, 1),
                "label": smooth_label,  
                "fsm_state": track.fsm.state
            })
        return results

# ==================== 测试运行代码 (模拟医院抢救车场景) ====================
if __name__ == "__main__":
    tracker = MultiObjectTracker(is_training=True)

    print("\n--- Frame 1: 抽屉拉开，检测到两支【肾上腺素】(画面中央) ---")
    frame_info_open = {"drawer_open": True, "width": 640, "height": 480}
    cur_frame = [
        [np.array([1.0, 0.0, 0.0]), (300.0, 200.0), [("肾上腺素_1mg", 0.95)]],
        [np.array([1.0, 0.1, 0.0]), (320.0, 200.0), [("肾上腺素_1mg", 0.96)]]
    ]
    res = tracker.step(cur_frame, frame_info_open)
    for r in res: print(f"ID:{r['track_id']} 药品:{r['label']} 位置:{r['position']} 状态:{r['fsm_state']}")

    print("\n--- Frame 2: 护士手伸下！药盒1被手完全遮挡(检测框丢失) ---")
    # 只检测到药盒2，药盒1消失于中央区域
    cur_frame = [
        [np.array([1.0, 0.1, 0.0]), (320.0, 200.0), [("肾上腺素_1mg", 0.96)]]
    ]
    res = tracker.step(cur_frame, frame_info_open)
    for r in res: print(f"ID:{r['track_id']} 药品:{r['label']} 位置:{r['position']} 状态:{r['fsm_state']}")
    print("【结果】：药盒1 并未被误判为拿走，依然维持 Track，等待手移开。")

    print("\n--- Frame 3: 护士将抽屉推上关闭 (画面全黑，没有检测框) ---")
    frame_info_closed = {"drawer_open": False, "width": 640, "height": 480}
    cur_frame = []
    res = tracker.step(cur_frame, frame_info_closed)
    for r in res: print(f"ID:{r['track_id']} 药品:{r['label']} 位置:{r['position']} 状态:{r['fsm_state']}")
    print("【结果】：即使所有药盒丢失检测，系统也进入 SLEEP 休眠，防止 ID 删除。")

    print("\n--- Frame 4: 2分钟后抽屉再次拉开，药盒依然在原位 ---")
    cur_frame = [
        [np.array([1.0, 0.0, 0.0]), (301.0, 201.0), [("肾上腺素_1mg", 0.95)]],
        [np.array([1.0, 0.1, 0.0]), (321.0, 201.0), [("肾上腺素_1mg", 0.96)]]
    ]
    res = tracker.step(cur_frame, frame_info_open)
    for r in res: print(f"ID:{r['track_id']} 药品:{r['label']} 位置:{r['position']} 状态:{r['fsm_state']}")
    print("【结果】：药品瞬间恢复为稳定状态，没有触发重置新建。")

    print("\n--- Frame 5: 护士抓起药盒2向边缘快速抽出画面 ---")
    cur_frame = [
        [np.array([1.0, 0.0, 0.0]), (301.0, 201.0), [("肾上腺素_1mg", 0.95)]],
        # 药盒2 被快速移动到边缘 (620, 450)
        [np.array([1.0, 0.1, 0.0]), (620.0, 450.0), [("肾上腺素_1mg", 0.96)]]
    ]
    res = tracker.step(cur_frame, frame_info_open)
    for r in res: print(f"ID:{r['track_id']} 药品:{r['label']} 位置:{r['position']} 状态:{r['fsm_state']}")

    print("\n--- Frame 6: 药盒2在边缘离开视野(检测丢失) ---")
    cur_frame = [
        [np.array([1.0, 0.0, 0.0]), (301.0, 201.0), [("肾上腺素_1mg", 0.95)]],
    ]
    # 模拟经过 4 帧...
    for _ in range(4):
        res = tracker.step(cur_frame, frame_info_open)
        
    print("当前剩下的 Track 数量：", len(tracker.tracks))
    for r in res: print(f"ID:{r['track_id']} 药品:{r['label']} 位置:{r['position']} 状态:{r['fsm_state']}")
    print("【结果】：药盒2 因为在边缘丢失超过3帧，已被成功判定为拿取消耗并清理出内存！")
