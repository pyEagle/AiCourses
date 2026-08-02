import numpy as np
from scipy.optimize import linear_sum_assignment

# ==================== 1. 状态定义与 FSM（有限状态机） ====================
class ObjectState:
    PLACED = "放入"
    STABLE = "稳定"
    SLOW_MOVING = "低速移动"
    FAST_MOVING = "快速移动"
    TAKEN_AWAY = "拿走"

class FiniteStateMachine:
    def __init__(self):
        self.state = ObjectState.PLACED
        self.missing_frames = 0
        self.age = 0

    def update(self, matched, velocity):
        self.age += 1
        if not matched:
            self.missing_frames += 1
            if self.missing_frames > 5:
                self.state = ObjectState.TAKEN_AWAY
            return

        self.missing_frames = 0
        
        # 【改进点】：使用基于目标尺度的相对速度进行判断会更好。
        # 这里为了兼容无 bbox 宽高的场景，保留绝对数值，但在工业界应除以 bbox 对角线长度。
        speed = np.linalg.norm(velocity)
        
        if self.age == 1:
            self.state = ObjectState.PLACED
        elif speed < 1.0:
            self.state = ObjectState.SLOW_MOVING
        elif 1.0 <= speed < 15.0:
            self.state = ObjectState.STABLE
        else:
            self.state = ObjectState.FAST_MOVING


# ==================== 2. 自适应卡尔曼滤波器 ====================
class AdaptiveKalmanFilter:
    def __init__(self, initial_pos):
        self.kf_dim = 4
        # 状态向量: [x, y, vx, vy]
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)
        
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
                           
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
                           
        self.P = np.eye(self.kf_dim) * 10.0
        self.R = np.eye(2) * 2.0
        self.Q = np.eye(self.kf_dim) * 0.1

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:2]

    def update(self, measurement, state):
        if state == ObjectState.FAST_MOVING:
            self.Q = np.eye(self.kf_dim) * 1.5
            self.R = np.eye(2) * 5.0
        elif state == ObjectState.STABLE:
            self.Q = np.eye(self.kf_dim) * 0.05
            self.R = np.eye(2) * 1.0
        else:
            self.Q = np.eye(self.kf_dim) * 0.1
            self.R = np.eye(2) * 2.0

        y = measurement - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.kf_dim)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(K, np.dot(self.R, K.T))

    # 【改进点 1】：新增马氏距离计算方法。具备尺度不变性，是 MOT 的标准空间度量。
    def get_mahalanobis_distance(self, measurement):
        y = measurement - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # 使用 solve 替代 inv，数值稳定性更好
        return np.sqrt(np.dot(y.T, np.linalg.solve(S, y)))


# ==================== 3. 维特比算法 (Viterbi) 标签平滑 ====================
def viterbi_decode(label_history):
    if not label_history:
        return None

    states = set()
    for topk in label_history:
        for label, _ in topk:
            states.add(label)
    states = list(states)
    if not states:
        return None

    num_states = len(states)
    T = len(label_history)
    
    viterbi_table = np.zeros((T, num_states))
    backpointer = np.zeros((T, num_states), dtype=int)

    # 【改进点 2】：理论上应从业务数据集的混淆矩阵加载，这里模拟一个合理的先验：
    # 类别维持不变的概率极大（0.95），发生状态跳变的概率被均匀分配到其他类别。
    trans_p = np.full((num_states, num_states), 0.05 / max(1, num_states - 1))
    np.fill_diagonal(trans_p, 0.95)

    first_topk = dict(label_history[0])
    for s_idx, state in enumerate(states):
        emission = first_topk.get(state, 0.01)
        viterbi_table[0, s_idx] = np.log(max(emission, 1e-5))

    for t in range(1, T):
        curr_topk = dict(label_history[t])
        for s_idx, state in enumerate(states):
            emission = curr_topk.get(state, 0.01)
            log_emission = np.log(max(emission, 1e-5))
            
            probabilities = viterbi_table[t-1] + np.log(trans_p[:, s_idx])
            best_prev_idx = np.argmax(probabilities)
            
            viterbi_table[t, s_idx] = probabilities[best_prev_idx] + log_emission
            backpointer[t, s_idx] = best_prev_idx

    best_last_state_idx = np.argmax(viterbi_table[T-1])
    best_path = [best_last_state_idx]
    for t in range(T-1, 0, -1):
        best_path.insert(0, backpointer[t, best_path[0]])

    return states[best_path[-1]]


# ==================== 4. 跟踪目标类 (Tracklet) ====================
class Track:
    def __init__(self, target_id, embedding, pos, label_topk3):
        self.track_id = target_id
        self.kf = AdaptiveKalmanFilter(pos)
        self.fsm = FiniteStateMachine()
        self.current_embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        self.embedding_history = [self.current_embedding]
        self.label_history = [label_topk3]
        self.current_pos = pos

    def update(self, embedding, pos, label_topk3):
        # 【改进点 3】：直接使用 KF 后验状态中的速度，摒弃极易产生高频噪声的帧间坐标差分
        velocity = self.kf.x[2:4] 
        
        self.current_pos = pos
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-6)
        
        # 计算当前外观与历史特征的相似度
        cos_sim = np.dot(self.current_embedding, norm_emb)
        
        # 【改进点 4】：动态 EMA。如果相似度低（目标可能被遮挡或严重畸变），
        # 则 alpha 逼近 0.95，强烈拒绝污染；若相似度高，则 alpha 降低，平滑吸收新特征。
        alpha = 0.95 - 0.15 * max(0, cos_sim)
        
        self.current_embedding = alpha * self.current_embedding + (1.0 - alpha) * norm_emb
        self.current_embedding /= (np.linalg.norm(self.current_embedding) + 1e-6)
        
        # KF 更新流程
        self.kf.predict()
        self.kf.update(np.array(pos), self.fsm.state)
        
        # 传递真实估计速度更新 FSM
        self.fsm.update(matched=True, velocity=velocity)
        
        self.embedding_history.append(self.current_embedding)
        self.label_history.append(label_topk3)
        
        # 保持 9 帧滑动窗口
        if len(self.label_history) > 9:
            self.label_history.pop(0)
            self.embedding_history.pop(0)

    def miss(self):
        self.kf.predict()
        self.fsm.update(matched=False, velocity=np.array([0.0, 0.0]))


# ==================== 5. 多目标跟踪主类 ====================
class MultiObjectTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 0

    def step(self, cur_frame):
        # 1. 卡尔曼预测
        predicted_positions = []
        for track in self.tracks:
            pred_pos = track.kf.predict()
            predicted_positions.append(pred_pos)

        num_tracks = len(self.tracks)
        num_dets = len(cur_frame)
        
        if num_tracks > 0 and num_dets > 0:
            cost_matrix = np.full((num_tracks, num_dets), 1e5) # 默认极高代价
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(cur_frame):
                    det_emb, det_pos, _ = det
                    norm_det_emb = det_emb / (np.linalg.norm(det_emb) + 1e-6)
                    
                    # 空间马氏距离
                    d_m = track.kf.get_mahalanobis_distance(det_pos)
                    
                    # 空间门控：卡方分布 0.95 置信度下 2 个自由度的阈值为 9.48。超出则认为不可能匹配。
                    if d_m > 9.48:
                        continue 
                    
                    # 外观余弦距离
                    cos_sim = np.dot(track.current_embedding, norm_det_emb)
                    cost_appearance = 1.0 - cos_sim
                    
                    # 【改进点 5】：使用平滑的指数函数实现空间门控权重分配，避免代价突变引发震荡。
                    # 距离越近（d_m 趋于0），w_spa 越趋近 0.8；距离越远，w_spa 平滑衰减到 0.4 左右。
                    w_spa = 0.4 + 0.4 * np.exp(-d_m / 2.0)
                    w_app = 1.0 - w_spa
                    
                    # 综合代价矩阵
                    spatial_norm = d_m / 9.48
                    cost_matrix[i, j] = w_app * cost_appearance + w_spa * spatial_norm

            # 匈牙利算法最优分配
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)

        matched_tracks = set()
        matched_dets = set()
        
        for r, c in zip(row_ind, col_ind):
            # 过滤掉被 gating 机制（值为1e5）或者综合代价过高的错误匹配
            if cost_matrix[r, c] < 0.8:
                matched_tracks.add(r)
                matched_dets.add(c)
                det_emb, det_pos, det_labels = cur_frame[c]
                self.tracks[r].update(det_emb, det_pos, det_labels)

        # 处理 Miss 与 New Detect
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.miss()

        for j, det in enumerate(cur_frame):
            if j not in matched_dets:
                det_emb, det_pos, det_labels = det
                new_track = Track(self.next_id, det_emb, det_pos, det_labels)
                self.next_id += 1
                self.tracks.append(new_track)

        # 垃圾回收
        self.tracks = [t for t in self.tracks if t.fsm.state != ObjectState.TAKEN_AWAY]

        # Viterbi 解码输出
        results = []
        for track in self.tracks:
            optimal_label = viterbi_decode(track.label_history)
            results.append({
                "track_id": track.track_id,
                "position": track.current_pos,
                "state": track.fsm.state,
                "reliable_label": optimal_label
            })

        return results


# ==================== 6. 测试运行代码 ====================
if __name__ == "__main__":
    tracker = MultiObjectTracker()

    # 第 1 帧：目标 A 放入画面
    cur_frame_1 = [
        [np.array([1.0, 0.0, 0.0]), (100.0, 200.0), [("apple", 0.95), ("orange", 0.04), ("pear", 0.01)]]
    ]
    print("--- Frame 1 (放入目标) ---")
    print(tracker.step(cur_frame_1))

    # 第 2 帧：目标 A 快速移动
    cur_frame_2 = [
        [np.array([0.98, 0.02, 0.0]), (130.0, 240.0), [("apple", 0.90), ("orange", 0.08), ("pear", 0.02)]]
    ]
    print("\n--- Frame 2 (目标快速移动) ---")
    print(tracker.step(cur_frame_2))

    # 第 3 帧：目标 A 保持稳定，目标 B 放入
    cur_frame_3 = [
        [np.array([0.97, 0.03, 0.0]), (132.0, 242.0), [("apple", 0.96), ("orange", 0.03), ("pear", 0.01)]],
        [np.array([0.0, 1.0, 0.0]), (500.0, 500.0), [("banana", 0.92), ("mango", 0.07), ("lemon", 0.01)]]
    ]
    print("\n--- Frame 3 (目标稳定，新增目标) ---")
    print(tracker.step(cur_frame_3))

    # 第 4 帧：目标 A 特征畸变！
    # 模拟远距离畸变，向量变为 [0.5, 0.5, 0.5]，但其位置 (134, 244) 在马氏距离(d_m)判定下依然完全合理。
    # 动态 EMA 会将 alpha 拉高，拒绝严重劣质特征；平滑门控会高度信任马氏距离匹配。
    cur_frame_4 = [
        [np.array([0.5, 0.5, 0.5]), (134.0, 244.0), [("apple", 0.60), ("orange", 0.35), ("pear", 0.05)]],
        [np.array([0.0, 0.98, 0.02]), (502.0, 501.0), [("banana", 0.94), ("mango", 0.05), ("lemon", 0.01)]]
    ]
    print("\n--- Frame 4 (抗特征畸变测试) ---")
    print(tracker.step(cur_frame_4))
