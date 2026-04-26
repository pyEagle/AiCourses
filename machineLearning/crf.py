# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

class TorchLiteCRFPro(nn.Module):
    def __init__(self, num_tags, device=None):
        super().__init__()
        self.num_tags = num_tags
        self.device = device if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        self.to(self.device)

    def _log_sum_exp(self, tensor, dim):
        return torch.logsumexp(tensor, dim)

    # ==========================================================
    # 1. 前向算法 (logZ)
    # emissions: [batch, seq_len, num_tags]
    # mask:      [batch, seq_len]
    # ==========================================================
    def _compute_log_partition(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape

        # alpha 初始化
        alpha = self.start_transitions + emissions[:, 0]  # [B, K]

        for t in range(1, seq_len):
            emit_t = emissions[:, t].unsqueeze(1)          # [B, 1, K]
            trans = self.transitions.unsqueeze(0)          # [1, K, K]

            score = alpha.unsqueeze(2) + trans + emit_t    # [B, K, K]

            new_alpha = torch.logsumexp(score, dim=1)      # [B, K]

            # mask 控制 padding
            mask_t = mask[:, t].unsqueeze(1)
            alpha = new_alpha * mask_t + alpha * (1 - mask_t)

        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=1)  # [B]

    # ==========================================================
    # 2. 计算路径 score
    # tags: [batch, seq_len]
    # ==========================================================
    def _compute_score(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.shape

        emit_score = emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1)
        trans_score = self.transitions[tags[:, :-1], tags[:, 1:]]

        start_score = self.start_transitions[tags[:, 0]]
        end_score = self.end_transitions[tags.gather(1, mask.sum(1).long().unsqueeze(1)-1).squeeze()]

        score = start_score + end_score
        score += (emit_score * mask).sum(dim=1)
        score += (trans_score * mask[:, 1:]).sum(dim=1)

        return score

    # ==========================================================
    # 3. 负对数似然 (训练目标)
    # ==========================================================
    def neg_log_likelihood(self, emissions, tags, mask):
        log_Z = self._compute_log_partition(emissions, mask)
        path_score = self._compute_score(emissions, tags, mask)
        return (log_Z - path_score).mean()

    # ==========================================================
    # 4. Viterbi 解码
    # ==========================================================
    def decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape

        dp = self.start_transitions + emissions[:, 0]  # [B, K]
        backpointers = []

        for t in range(1, seq_len):
            score = dp.unsqueeze(2) + self.transitions.unsqueeze(0)  # [B, K, K]
            best_score, best_path = torch.max(score, dim=1)  # [B, K]
            dp = best_score + emissions[:, t]
            backpointers.append(best_path)

        dp = dp + self.end_transitions
        best_last_score, best_last_tag = torch.max(dp, dim=1)

        # 回溯
        best_paths = []
        for b in range(batch_size):
            seq_len_b = int(mask[b].sum().item())
            last_tag = best_last_tag[b].item()

            path = [last_tag]
            for bp in reversed(backpointers[:seq_len_b-1]):
                last_tag = bp[b][last_tag].item()
                path.append(last_tag)

            path.reverse()
            best_paths.append(path)

        return best_paths

if __name__ == "__main__":
    torch.manual_seed(0)

    B = 2
    T = 5
    K = 3

    crf = TorchLiteCRFPro(num_tags=K)

    # 模拟 emission（通常来自 BiLSTM / Transformer）
    emissions = torch.randn(B, T, K).to(crf.device)

    # 标签
    tags = torch.tensor([
        [0,1,2,1,0],
        [2,2,1,0,0]
    ], device=crf.device)

    # mask（1表示有效位置）
    mask = torch.tensor([
        [1,1,1,1,1],
        [1,1,1,0,0]
    ], dtype=torch.float32, device=crf.device)

    # 优化器
    optimizer = torch.optim.Adam(crf.parameters(), lr=0.01)

    print("=== Training ===")
    for epoch in range(20):
        loss = crf.neg_log_likelihood(emissions, tags, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    print("\n=== Decoding ===")
    paths = crf.decode(emissions, mask)
    print("Best paths:", paths)

