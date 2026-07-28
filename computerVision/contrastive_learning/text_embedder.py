import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


class TextProjector(nn.Module):
    """与训练脚本完全一致的文本投影层"""
    def __init__(self, in_features=768, out_features=256):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        out = self.proj(x)
        return F.normalize(out, p=2, dim=1)


class TextEmbedder:
    def __init__(self, projector_weight_path, device=None):
        """
        初始化文本嵌入提取器
        :param projector_weight_path: 训练保存的权重文件路径
        :param device: 计算设备，默认自动选择cuda/cpu
        """
        # 设备自动选择
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 加载中文BERT与分词器
        print("正在加载中文BERT模型与分词器...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.text_encoder = BertModel.from_pretrained('bert-base-chinese').to(self.device)
        self.text_encoder.eval()

        # 初始化并加载文本投影器
        self.text_projector = TextProjector(in_features=768, out_features=256).to(self.device)
        self._load_projector_weights(projector_weight_path)
        self.text_projector.eval()

        print(f"✅ TextEmbedder 初始化完成，运行设备: {self.device}")

    def _load_projector_weights(self, weight_path):
        """加载文本投影器权重，兼容训练脚本的checkpoint格式"""
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"找不到权重文件: {weight_path}")

        # 修复安全警告：开启 weights_only，仅加载张量数据
        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=True)

        # 优先匹配训练脚本保存的嵌套字典格式
        if isinstance(checkpoint, dict) and "text_projector" in checkpoint:
            proj_state = checkpoint["text_projector"]
        else:
            # 兼容单独保存 text_projector 权重的情况
            proj_state = checkpoint

        self.text_projector.load_state_dict(proj_state)

    def text_embedding(self, text):
        """
        提取单个文本的256维嵌入向量
        :param text: 输入文本字符串
        :return: numpy数组，shape=(256,)
        """
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            bert_out = self.text_encoder(**encoded)
            cls_feat = bert_out.last_hidden_state[:, 0, :]
            final_emb = self.text_projector(cls_feat)

        return final_emb.cpu().numpy()[0]

    def batch_text_embedding(self, text_list, save_flag=False, save_dir='./text_embeddings'):
        """
        批量提取文本嵌入向量
        :param text_list: 文本列表
        :param save_flag: 是否将结果保存到本地
        :param save_dir: 保存目录路径
        :return: 字典 {文本字符串: 嵌入向量numpy数组}
        """
        result_dict = {}
        for text in text_list:
            emb = self.text_embedding(text)
            result_dict[text] = emb
            #print(text, emb)
            print(text)

        if save_flag:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'text_embeddings.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(result_dict, f)
            print(f"✅ 批量文本嵌入已保存至: {save_path}")

        return result_dict


if __name__ == "__main__":
    # ========== 示例用法 ==========
    weight_path = "./saved_weights/medicine_embedder_best_rknn_opt.pth"
    embedder = TextEmbedder(weight_path)

    import sys
    in_dir = sys.argv[1]
    name = set()
    for f in os.listdir(in_dir):
        item = f.split('_')[2][-7:]
        name.add(item)

    text_list = list(name)
    emb_dict = embedder.batch_text_embedding(
        text_list,
        save_flag=True,
        save_dir="./text_embeddings"
    )
    print(f"\n批量处理完成，共生成 {len(emb_dict)} 个文本嵌入")
