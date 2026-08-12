import numpy as np
from collections import deque
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class TrackerConfig:
    def __init__(self, config_path="tracker_cfg.json", auto_load=True):
        self.config_path = config_path
        self.feature_dim = 3
        self.id_ring_max = 9999
        self.max_ttl = 3600.0
        self.covered_timeout = 1.0
        self.hidden_base_timeout = 0.3
        self.pending_delete_confirm = 0.5
        self.missing_edge_fast = 0.1
        self.missing_edge_slow = 0.5
        self.missing_center = 1.0
        self.edge_margin_ratio = 0.10
        self.gallery_max_size = 5
        self.gallery_update_thresh = 0.90
        self.gallery_ema_alpha = 0.1
        self.gate_mah_normal = 9.48
        self.gate_mah_wakeup = 15.0
        self.gate_cos = 0.5
        self.kf_r_weight = 5.0    
        self.kf_q_weight = 100.0  
        if auto_load:
            self._load_from_file()

    def _load_from_file(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if hasattr(self, k):
                            setattr(self, k, v)
            except Exception:
                pass

class ObjectState:
    PLACED = "放入"
    STABLE = "稳定在抽屉中"
    SLOW_MOVING = "碰触/微移"
    FAST_MOVING = "快速拿取/放入"
    SLEEP = "休眠"
    COVERED = "被遮挡"
    HIDDEN_BY_CABINET = "藏于柜内"
    PENDING_DELETE = "待删除确认"
    TAKEN_AWAY = "拿走"

class OnlineSpeedGMM:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.means = np.array([15.0, 150.0, 450.0], dtype=float)
        self.variances = np.array([25.0, 400.0, 2500.0], dtype=float)
        self.weights = np.array([0.33, 0.33, 0.34], dtype=float)
        self.warmup_buffer = [10.0, 15.0, 140.0, 160.0, 400.0, 500.0]
        self.warmup_size = 30
        self.is_warmed_up = False

    def update_and_predict(self, speed):
        speed = max(0.0, float(speed))
        if not self.is_warmed_up:
            self.warmup_buffer.append(speed)
            if len(self.warmup_buffer) >= self.warmup_size:
                data = np.array(self.warmup_buffer, dtype=float)
                p33, p66 = np.percentile(data, [33, 66])
                g0, g1, g2 = data[data < p33], data[(data >= p33) & (data < p66)], data[data >= p66]
                if len(g0): self.means[0], self.variances[0] = np.mean(g0), max(1.0, np.var(g0))
                if len(g1): self.means[1], self.variances[1] = np.mean(g1), max(1.0, np.var(g1))
                if len(g2): self.means[2], self.variances[2] = np.mean(g2), max(1.0, np.var(g2))
                self.is_warmed_up = True
            if speed < 30.0: return 0
            elif speed < 240.0: return 1
            else: return 2

        stds = np.sqrt(self.variances) + 1e-6
        pdfs = self.weights * (1.0 / (np.sqrt(2 * np.pi) * stds)) * np.exp(-0.5 * ((speed - self.means) / stds)**2)
        max_idx, max_pdf = int(np.argmax(pdfs)), np.max(pdfs)
        
        if max_pdf < 1e-3: max_idx = int(np.argmin(np.abs(self.means - speed)))

        self.weights = (1 - self.alpha) * self.weights
        self.weights[max_idx] += self.alpha
        self.weights = np.maximum(self.weights, 0.05)
        self.weights /= np.sum(self.weights)
        
        old_mean = self.means[max_idx]
        self.means[max_idx] = (1 - self.alpha) * old_mean + self.alpha * speed
        
        diff_sq = min((speed - old_mean)**2, self.variances[max_idx] * 25.0)
        self.variances[max_idx] = max(1.0, (1 - self.alpha) * self.variances[max_idx] + self.alpha * diff_sq)
        self.means[max_idx] = np.clip(self.means[max_idx], [0, 60, 300][max_idx], [45, 240, 900][max_idx])

        if speed < 30.0: return 0
        return max_idx

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0]-boxA[2]/2, boxB[0]-boxB[2]/2), max(boxA[1]-boxA[3]/2, boxB[1]-boxB[3]/2)
    xB, yB = min(boxA[0]+boxA[2]/2, boxB[0]+boxB[2]/2), min(boxA[1]+boxA[3]/2, boxB[1]+boxB[3]/2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    return interArea / float(boxA[2]*boxA[3] + boxB[2]*boxB[3] - interArea)

def smooth_label(label_history):
    if not label_history: return None
    scores = {}
    for topk in label_history:
        for label, conf in topk: scores[label] = scores.get(label, 0.0) + conf
    return max(scores.items(), key=lambda x: x[1])[0]

def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-6) + xi)
        if intersect:
            inside = not inside
    return inside

def fast_greedy_assignment(cost_matrix):
    if cost_matrix.size == 0:
        return [], []
    cost_flat = cost_matrix.flatten()
    indices = np.argsort(cost_flat)
    used_rows, used_cols = set(), set()
    row_ind, col_ind = [], []
    num_cols = cost_matrix.shape[1]
    for idx in indices:
        r, c = divmod(idx, num_cols)
        if r not in used_rows and c not in used_cols:
            row_ind.append(r)
            col_ind.append(c)
            used_rows.add(r)
            used_cols.add(c)
    return row_ind, col_ind

class FeatureGallery:
    def __init__(self, initial_emb, config: TrackerConfig):
        self.cfg = config
        norm_emb = initial_emb / (np.linalg.norm(initial_emb) + 1e-6)
        self.gallery = [norm_emb]

    def update(self, emb, max_sim, best_idx, visibility=1.0):
        if visibility < 0.8: return
        norm_emb = emb / (np.linalg.norm(emb) + 1e-6)
        if max_sim < self.cfg.gallery_update_thresh:
            if len(self.gallery) >= self.cfg.gallery_max_size:
                self.gallery.pop(1)
            self.gallery.append(norm_emb)
        else:
            alpha = self.cfg.gallery_ema_alpha
            self.gallery[best_idx] = (1.0 - alpha) * self.gallery[best_idx] + alpha * norm_emb
            self.gallery[best_idx] /= (np.linalg.norm(self.gallery[best_idx]) + 1e-6)

class KinematicKalmanFilter:
    def __init__(self, initial_pos, config: TrackerConfig):
        self.cfg = config
        self.kf_dim = 4
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.P = np.eye(self.kf_dim) * 10.0
        self.R = np.eye(2) * self.cfg.kf_r_weight
        self._eps = 1e-6
        self.adaptive_q_scale = 1.0

    def predict(self, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        sigma_a_sq = self.cfg.kf_q_weight * self.adaptive_q_scale
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        Q = sigma_a_sq * np.array([
            [dt4/4, 0, dt3/2, 0], [0, dt4/4, 0, dt3/2], [dt3/2, 0, dt2, 0], [0, dt3/2, 0, dt2]
        ], dtype=float)
        self.x = np.dot(F, self.x)
        self.P = np.dot(np.dot(F, self.P), F.T) + Q
        self.P = (self.P + self.P.T) / 2.0
        return self.x[:2].copy()

    def update(self, measurement):
        y = np.array(measurement) - np.dot(self.H, self.x)
        self.adaptive_q_scale = np.clip(np.linalg.norm(y) / 5.0, 1.0, 10.0)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R + np.eye(2) * self._eps
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.kf_dim)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(K, np.dot(self.R, K.T))
        self.P = (self.P + self.P.T) / 2.0
        self.P = np.clip(self.P, 0, 1000.0)

    def get_mahalanobis_distance(self, measurements):
        Y = measurements - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R + np.eye(2) * self._eps
        S_inv = np.linalg.inv(S)
        return np.sqrt(np.maximum(0, np.sum(np.dot(Y, S_inv) * Y, axis=1)))

class FiniteStateMachine:
    def __init__(self, global_gmm, config: TrackerConfig):
        self.cfg = config
        self.state = ObjectState.PLACED
        self.missing_time, self.pending_time, self.covered_time, self.hidden_time = 0.0, 0.0, 0.0, 0.0
        self.hits = 1
        self.speed_gmm = global_gmm
        self.last_pos = None

    def update(self, matched, velocity, pos, dt, frame_info, is_covered=False):
        poly = frame_info.get("drawer_polygon", None)
        if poly is None:
            xmin, ymin, xmax, ymax = frame_info.get("drawer_bbox", [0, 0, 640, 480])
            poly = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            
        diag_len = max(100.0, np.linalg.norm(np.array(poly[0]) - np.array(poly[2])))

        if not matched:
            if is_covered:
                self.state = ObjectState.COVERED
                self.covered_time += dt
                self.missing_time = 0.0
                if self.covered_time > self.cfg.covered_timeout:
                    self.state, self.pending_time = ObjectState.PENDING_DELETE, 0.0
                return
            self.missing_time += dt
            x, y = self.last_pos if self.last_pos is not None else pos
            if not point_in_polygon([x, y], poly):
                self.state = ObjectState.HIDDEN_BY_CABINET
                self.hidden_time += dt
                self.missing_time = 0.0
                open_ratio = float(np.clip(frame_info.get("drawer_open_ratio", 1.0), 0.05, 1.0))
                if self.hidden_time > self.cfg.hidden_base_timeout + (1.0 - open_ratio) * 1.5:
                    self.state, self.pending_time = ObjectState.PENDING_DELETE, 0.0
                return
            tol_time = self.cfg.missing_center
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
        self.missing_time, self.pending_time = 0.0, 0.0
        self.hits += 1
        speed = np.linalg.norm(velocity)
        standard_speed = (speed / diag_len) * 800.0
        speed_rank = self.speed_gmm.update_and_predict(standard_speed)
        
        if self.hits < 3: self.state = ObjectState.PLACED
        elif speed_rank == 0: self.state = ObjectState.STABLE
        elif speed_rank == 1: self.state = ObjectState.SLOW_MOVING
        else: self.state = ObjectState.FAST_MOVING

class Track:
    def __init__(self, target_id, embedding, pos, size, label_topk3, global_gmm, config: TrackerConfig):
        self.track_id = target_id
        self.cfg = config
        self.kf = KinematicKalmanFilter(pos, config)
        self.fsm = FiniteStateMachine(global_gmm, config)
        self.gallery = FeatureGallery(embedding, config)
        self.label_history = deque([label_topk3], maxlen=9)
        self.current_pos, self.size = np.array(pos, dtype=float), np.array(size, dtype=float)
        self.inactive_time = 0.0

    def update(self, embedding, pos, size, max_sim, best_idx, visibility, dt, frame_info, label_topk3):
        self.current_pos = np.array(pos, dtype=float)
        self.size = 0.8 * self.size + 0.2 * np.array(size, dtype=float)
        self.gallery.update(embedding, max_sim, best_idx, visibility)
        velocity_pred = self.kf.x[2:4].copy()
        self.fsm.update(matched=True, velocity=velocity_pred, pos=pos, dt=dt, frame_info=frame_info)
        self.kf.update(self.current_pos)
        self.label_history.append(label_topk3)
        self.inactive_time = 0.0

    def miss(self, dt, frame_info, is_covered=False):
        pred_pos = self.kf.x[:2].copy()
        self.current_pos = pred_pos
        self.fsm.update(matched=False, velocity=self.kf.x[2:4], pos=pred_pos, dt=dt, frame_info=frame_info, is_covered=is_covered)
        self.inactive_time += dt

class MultiObjectTracker:
    def __init__(self, config=None):
        self.cfg = config if config else TrackerConfig()
        self.tracks = []
        self.next_id = 0
        self.global_speed_gmm = OnlineSpeedGMM(alpha=0.15)

    def step(self, cur_frame, frame_info=None, dt=0.033):
        if frame_info is None: frame_info = {"drawer_bbox": [0,0,640,480], "drawer_open_ratio": 1.0}
        num_dets = len(cur_frame)
        det_embs, det_centers, det_sizes, det_labels, det_visibilities = [], [], [], [], []
        
        for det in cur_frame:
            emb, (x1, y1, x2, y2), labels = det[:3]
            vis = det[3] if len(det) > 3 else 1.0
            w, h = max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))
            cx, cy = float(x1 + x2) / 2.0, float(y1 + y2) / 2.0
            norm_emb = emb / (np.linalg.norm(emb) + 1e-6)
            det_embs.append(norm_emb)
            det_centers.append([cx, cy])
            det_sizes.append([w, h])
            det_labels.append(labels)
            det_visibilities.append(vis)
            
        feature_dim = len(cur_frame[0][0]) if num_dets > 0 else self.cfg.feature_dim
        det_embs = np.array(det_embs) if num_dets > 0 else np.empty((0, feature_dim))
        det_centers = np.array(det_centers) if num_dets > 0 else np.empty((0, 2))
        
        for track in self.tracks:
            if track.fsm.state in [ObjectState.COVERED, ObjectState.HIDDEN_BY_CABINET, ObjectState.PENDING_DELETE]:
                track.kf.x[2:4], track.kf.P = 0.0, np.clip(track.kf.P + np.eye(4)*10.0, 0, 500.0)
            track.kf.predict(dt)
            
        num_tracks = len(self.tracks)
        gate_mah = self.cfg.gate_mah_normal
        matched_tracks, matched_dets = set(), set()
        
        if num_tracks > 0 and num_dets > 0:
            active_states = [ObjectState.STABLE, ObjectState.SLOW_MOVING, ObjectState.FAST_MOVING, ObjectState.PLACED]
            active_indices = [i for i, t in enumerate(self.tracks) if t.fsm.state in active_states]
            
            if active_indices:
                cost_matrix = np.full((len(active_indices), num_dets), 1e5)
                meta_data = {}
                for row_idx, track_idx in enumerate(active_indices):
                    track = self.tracks[track_idx]
                    d_m_array = track.kf.get_mahalanobis_distance(det_centers)
                    track_sims = np.dot(np.array(track.gallery.gallery), det_embs.T)
                    max_sims = np.max(track_sims, axis=0)
                    best_idxs = np.argmax(track_sims, axis=0)
                    
                    for col_idx in range(num_dets):
                        d_m, max_sim = d_m_array[col_idx], max_sims[col_idx]
                        if d_m > gate_mah or max_sim < self.cfg.gate_cos: continue
                        w_spa = 0.7 * np.exp(-d_m / 3.0)
                        cost_matrix[row_idx, col_idx] = (1.0 - w_spa)*(1.0 - max_sim) + w_spa*(d_m / gate_mah)
                        meta_data[(row_idx, col_idx)] = (max_sim, best_idxs[col_idx])
                        
                row_ind, col_ind = fast_greedy_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < 0.8:
                        matched_tracks.add(active_indices[r])
                        matched_dets.add(c)
                        m_sim, b_idx = meta_data[(r, c)]
                        self.tracks[active_indices[r]].update(det_embs[c], det_centers[c], det_sizes[c], m_sim, b_idx, det_visibilities[c], dt, frame_info, det_labels[c])

            unmatched_tracks = [i for i in range(num_tracks) if i not in matched_tracks]
            unmatched_dets = [j for j in range(num_dets) if j not in matched_dets]
            
            if unmatched_tracks and unmatched_dets:
                cost_matrix = np.full((len(unmatched_tracks), len(unmatched_dets)), 1e5)
                meta_data = {}
                for row_idx, track_idx in enumerate(unmatched_tracks):
                    track = self.tracks[track_idx]
                    unmatched_det_centers = det_centers[unmatched_dets]
                    d_m_array = track.kf.get_mahalanobis_distance(unmatched_det_centers)
                    track_sims = np.dot(np.array(track.gallery.gallery), det_embs[unmatched_dets].T)
                    max_sims = np.max(track_sims, axis=0)
                    best_idxs = np.argmax(track_sims, axis=0)
                    relaxed_gate_mah = gate_mah * 1.5
                    relaxed_gate_cos = self.cfg.gate_cos * 0.8
                    
                    for sub_col_idx, col_idx in enumerate(unmatched_dets):
                        d_m, max_sim = d_m_array[sub_col_idx], max_sims[sub_col_idx]
                        if d_m > relaxed_gate_mah or max_sim < relaxed_gate_cos: continue
                        w_spa = 0.8 * np.exp(-d_m / 5.0)
                        cost_matrix[row_idx, sub_col_idx] = (1.0 - w_spa)*(1.0 - max_sim) + w_spa*(d_m / relaxed_gate_mah)
                        meta_data[(row_idx, sub_col_idx)] = (max_sim, best_idxs[sub_col_idx])
                        
                row_ind, col_ind = fast_greedy_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < 1.5:
                        real_trk, real_det = unmatched_tracks[r], unmatched_dets[c]
                        matched_tracks.add(real_trk); matched_dets.add(real_det)
                        m_sim, b_idx = meta_data[(r, c)]
                        self.tracks[real_trk].update(det_embs[real_det], det_centers[real_det], det_sizes[real_det], m_sim, b_idx, det_visibilities[real_det], dt, frame_info, det_labels[real_det])

        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                is_covered = False
                pred_box = (track.current_pos[0], track.current_pos[1], track.size[0], track.size[1])
                for j in range(num_dets):
                    det_box = (det_centers[j][0], det_centers[j][1], det_sizes[j][0], det_sizes[j][1])
                    if compute_iou(pred_box, det_box) > 0.3:
                        is_covered = True; break
                track.miss(dt, frame_info, is_covered)

        for j in range(num_dets):
            if j not in matched_dets:
                self.tracks.append(Track(self.next_id, det_embs[j], det_centers[j], det_sizes[j], det_labels[j], self.global_speed_gmm, self.cfg))
                self.next_id = (self.next_id + 1) % self.cfg.id_ring_max

        self.tracks = [
            t for t in self.tracks
            if t.fsm.state != ObjectState.TAKEN_AWAY and t.inactive_time < self.cfg.max_ttl
        ]
        
        return [{"track_id": t.track_id, "position": np.round(t.current_pos, 1), "size": np.round(t.size, 1), "label": smooth_label(list(t.label_history)), "fsm_state": t.fsm.state} for t in self.tracks]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RolloutBuffer:
    def __init__(self):
        self.actions, self.states, self.logprobs = [], [], []
        self.rewards, self.is_terminals = [], []
    def clear(self):
        self.actions.clear(); self.states.clear(); self.logprobs.clear()
        self.rewards.clear(); self.is_terminals.clear()

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh() 
        )
        self.action_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        action_std = self.action_logstd.exp()
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_std = self.action_logstd.exp()
        dist = Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        return action_logprobs, state_values.squeeze(-1), dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device).unsqueeze(0)
            action, action_logprob = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        return action.cpu().numpy()[0]

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0), dim=1).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0), dim=1).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0), dim=1).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

class MOTParameterSearchEnv:
    def __init__(self):
        self.state_dim = 6 
        self.action_dim = 5
        self.step_count = 0
        self.max_steps = 5  
        self.last_reward = 0.0
        self.current_params = np.zeros(5, dtype=np.float32)

    def reset(self):
        self.step_count = 0
        self.last_reward = 0.0
        self.current_params = np.zeros(5, dtype=np.float32)
        return self._get_state()

    def _get_state(self):
        return np.array([*self.current_params, self.last_reward], dtype=np.float32)

    def _action_to_config(self, action):
        gate_mah_normal = 12.5 + action[0] * 7.5   
        gate_cos = 0.5 + action[1] * 0.3           
        covered_timeout = 1.3 + action[2] * 1.2    
        kf_r_weight = 10.5 + action[3] * 9.5
        kf_q_weight = 155.0 + action[4] * 145.0
        
        return {
            "gate_mah_normal": float(np.clip(gate_mah_normal, 5.0, 20.0)),
            "gate_cos": float(np.clip(gate_cos, 0.2, 0.8)),
            "covered_timeout": float(np.clip(covered_timeout, 0.1, 2.5)),
            "kf_r_weight": float(np.clip(kf_r_weight, 1.0, 20.0)),
            "kf_q_weight": float(np.clip(kf_q_weight, 10.0, 300.0))
        }

    def step(self, action):
        self.current_params = np.clip(action, -1.0, 1.0)
        config_dict = self._action_to_config(self.current_params)
        
        reward = self._simulate_offline_episode(config_dict)
        self.last_reward = reward
        self.step_count += 1
        
        done = self.step_count >= self.max_steps
        return self._get_state(), reward, done, config_dict

    def _simulate_offline_episode(self, config_dict):
        cfg = TrackerConfig(auto_load=False)
        for k, v in config_dict.items():
            setattr(cfg, k, v)
            
        tracker = MultiObjectTracker(cfg)
        frame_info = {"drawer_bbox": [0,0,640,480], "drawer_open_ratio": 1.0}
        
        tracked_frames, id_switches, last_id = 0, 0, -1
        total_steps = 50
        dt = 0.033
        
        for i in range(total_steps):
            if 20 <= i <= 25: 
                frame_data = [] 
            else:
                x = 100 + i * 5.0 + np.random.normal(0, 5.0 if cfg.gate_mah_normal > 10 else 15.0)
                y = 100 + i * 2.0
                emb = np.array([1.0, 0.5 - i*0.005, 0.1])
                frame_data = [[emb, (x, y, x+50, y+50), [("模拟物体", 0.9)], 1.0]]
                
            res = tracker.step(frame_data, frame_info, dt)
            if len(res) > 0:
                tracked_frames += 1
                cur_id = res[0]['track_id']
                if last_id != -1 and cur_id != last_id:
                    id_switches += 1
                last_id = cur_id
                
        reward = (tracked_frames / total_steps) - (id_switches * 0.3)
        return max(0.0, min(1.0, reward))

if __name__ == "__main__":
    print("\n" + "="*60)
    print(f"MOT: PPO 5D Optimization Engine (FSM + KF)")
    print(f"Device: {device.type.upper()}")
    print("="*60)
    
    env = MOTParameterSearchEnv()
    ppo_agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=0.002)
    
    max_episodes = 30           
    update_timestep = 15        
    
    time_step = 0
    best_reward = -1
    best_config = None
    
    for ep in range(1, max_episodes + 1):
        state = env.reset()
        ep_reward = 0
        
        while True:
            action = ppo_agent.select_action(state)
            state, reward, done, current_config = env.step(action)
            
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            
            time_step += 1
            ep_reward += reward
            
            if reward > best_reward:
                best_reward = reward
                best_config = current_config
            
            if time_step % update_timestep == 0:
                ppo_agent.update()
                
            if done:
                break
                
        if ep % 5 == 0:
            print(f"  [Epoch {ep:02d}/{max_episodes}] Avg Reward: {ep_reward/env.max_steps:.3f} | Best: {best_reward:.3f}")

    print(f"\nOptimization completed. Best Config:\n {json.dumps(best_config, indent=2)}")
    
    json_path = "tracker_cfg.json"
    with open(json_path, 'w') as f:
        json.dump(best_config, f, indent=4)
    print(f"Exported to {json_path}")
    
    print("\n[Online Deployment Simulation]")
    online_tracker = MultiObjectTracker() 
    print(f"Tracker initialized.")
    print(f"Loaded FSM -> gate_mah_normal: {online_tracker.cfg.gate_mah_normal:.3f}, covered_timeout: {online_tracker.cfg.covered_timeout:.3f}")
    print(f"Loaded KF  -> kf_r_weight: {online_tracker.cfg.kf_r_weight:.3f}, kf_q_weight: {online_tracker.cfg.kf_q_weight:.3f}")
