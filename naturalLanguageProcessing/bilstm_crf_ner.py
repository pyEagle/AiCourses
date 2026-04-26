# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"[Device] Using: {DEVICE}")


sentences = [
    "我爱北京天安门",
    "小明在上海工作"
]

tags = [
    ["O", "O", "B-LOC", "I-LOC", "I-LOC", "I-LOC", "I-LOC"],
    ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "O"]
]

word2id = {}
tag2id = {"START":0, "STOP":1}

for sent in sentences:
    for ch in sent:
        if ch not in word2id:
            word2id[ch] = len(word2id)

for tag_seq in tags:
    for t in tag_seq:
        if t not in tag2id:
            tag2id[t] = len(tag2id)

id2tag = {v:k for k,v in tag2id.items()}

VOCAB_SIZE = len(word2id)
TAG_SIZE = len(tag2id)


def encode_sentence(sent):
    return torch.tensor([word2id[ch] for ch in sent], dtype=torch.long, device=DEVICE)

def encode_tags(tag_seq):
    return torch.tensor([tag2id[t] for t in tag_seq], dtype=torch.long, device=DEVICE)


class CRF(nn.Module):
    def __init__(self, tag_size):
        super().__init__()
        self.tag_size = tag_size

        self.transitions = nn.Parameter(torch.randn(tag_size, tag_size))

        self.START = tag2id["START"]
        self.STOP = tag2id["STOP"]

        self.transitions.data[:, self.START] = -10000
        self.transitions.data[self.STOP, :] = -10000

    def log_sum_exp(self, vec):
        max_score = vec.max()
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score)))

    def forward_alg(self, emissions):
        alpha = torch.full((self.tag_size,), -10000., device=DEVICE)
        alpha[self.START] = 0

        for emit in emissions:
            next_alpha = torch.full((self.tag_size,), -10000., device=DEVICE)

            for next_tag in range(self.tag_size):
                scores = alpha + self.transitions[:, next_tag] + emit[next_tag]
                next_alpha[next_tag] = self.log_sum_exp(scores)

            alpha = next_alpha

        alpha = alpha + self.transitions[:, self.STOP]
        return self.log_sum_exp(alpha)

    def score_sentence(self, emissions, tags):
        score = torch.tensor(0., device=DEVICE)
        tags = torch.cat([torch.tensor([self.START], device=DEVICE), tags])

        for i, emit in enumerate(emissions):
            score += self.transitions[tags[i], tags[i+1]] + emit[tags[i+1]]

        score += self.transitions[tags[-1], self.STOP]
        return score

    def viterbi_decode(self, emissions):
        backpointers = []

        alpha = torch.full((self.tag_size,), -10000., device=DEVICE)
        alpha[self.START] = 0

        for emit in emissions:
            backptr = []
            next_alpha = []

            for next_tag in range(self.tag_size):
                scores = alpha + self.transitions[:, next_tag]
                best_tag = torch.argmax(scores)
                backptr.append(best_tag)
                next_alpha.append(scores[best_tag] + emit[next_tag])

            alpha = torch.stack(next_alpha)
            backpointers.append(backptr)

        alpha = alpha + self.transitions[:, self.STOP]
        best_last = torch.argmax(alpha)

        best_path = [best_last.item()]
        for backptr in reversed(backpointers):
            best_last = backptr[best_last]
            best_path.append(best_last.item())

        best_path.pop()  # 去掉START
        best_path.reverse()
        return best_path


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_size, emb_dim=64, hidden_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, tag_size)

        self.crf = CRF(tag_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence).unsqueeze(0)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.fc(lstm_out.squeeze(0))
        return emissions

    def loss(self, sentence, tags):
        emissions = self.forward(sentence)
        forward_score = self.crf.forward_alg(emissions)
        gold_score = self.crf.score_sentence(emissions, tags)
        return forward_score - gold_score

    def predict(self, sentence):
        emissions = self.forward(sentence)
        return self.crf.viterbi_decode(emissions)


model = BiLSTM_CRF(VOCAB_SIZE, TAG_SIZE).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.01)

EPOCHS = 30

for epoch in range(EPOCHS):
    total_loss = 0

    for sent, tag_seq in zip(sentences, tags):
        model.zero_grad()

        sent_tensor = encode_sentence(sent)
        tag_tensor = encode_tags(tag_seq)

        loss = model.loss(sent_tensor, tag_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")


test_sentence = "我在北京工作"
test_tensor = encode_sentence(test_sentence)

pred_ids = model.predict(test_tensor)
pred_tags = [id2tag[i] for i in pred_ids]

print("\n[Test]")
print("Sentence:", test_sentence)
print("Pred:", pred_tags)
