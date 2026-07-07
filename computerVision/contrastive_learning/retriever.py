import os
import sys
import argparse
import numpy as np
import faiss
import json
from PIL import Image
import onnxruntime as ort

class MedicineBoxONNXRetriever:
    def __init__(self, onnx_model_path: str, db_dir: str, index_dir: str = "./faiss_index"):
        self.db_dir = db_dir
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "embedding.index")
        self.meta_path = os.path.join(index_dir, "image_meta.json")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[ONNX] 模型加载成功，使用设备: {self.session.get_providers()[0]}")

        self.index = None       # Faiss索引对象
        self.image_meta = []    # 图片元数据列表
        self.input_size = (320, 320)

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        img = img.resize(self.input_size, resample=Image.BILINEAR)
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_np = img_np.transpose((2, 0, 1))

        return np.expand_dims(img_np, axis=0).astype(np.float32)

    def _generate_mirror_images(self, img_path: str) -> list:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"无法读取图片文件: {img_path}, 错误: {e}")

        mirror_list = [
            (img, "original"),                                                           # 原图
            (img.transpose(Image.FLIP_LEFT_RIGHT), "horizontal"),                        # 左右镜像
            (img.transpose(Image.FLIP_TOP_BOTTOM), "vertical"),                          # 上下镜像
            (img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM), "diagonal") # 对角线
        ]
        return mirror_list

    def get_embedding(self, image_input) -> np.ndarray:
        """提取特征并进行 L2 归一化，兼容多种输入格式"""
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                img = Image.fromarray(image_input[:, :, ::-1])
            else:
                img = Image.fromarray(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")

        input_data = self._preprocess(img)

        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        embedding = outputs[0].squeeze()  # 去掉batch维度，得到 (256,)

        norm = np.linalg.norm(embedding, ord=2)
        if norm > 1e-8:
            embedding = embedding / norm
        else:
            embedding = np.zeros_like(embedding)

        return embedding.astype(np.float32)

    def build_index(self, enable_mirror: bool = True):
        print(f"[构建索引] 开始遍历数据库目录: {self.db_dir}")
        print(f"[构建索引] 镜像增强: {'开启' if enable_mirror else '关闭'}")
        os.makedirs(self.index_dir, exist_ok=True)

        exts = (".jpg", ".jpeg", ".png", ".bmp")
        
        img_paths = []
        for root, _, files in os.walk(self.db_dir):
            for file in files:
                if file.lower().endswith(exts):
                    img_paths.append(os.path.join(root, file))

        if not img_paths:
            raise ValueError(f"目录 {self.db_dir} 及其子目录中未找到任何图片文件")

        emb_list = []
        self.image_meta = []

        for idx, img_path in enumerate(img_paths):
            img_name = os.path.basename(img_path)
            try:
                if enable_mirror:
                    mirror_list = self._generate_mirror_images(img_path)
                    for img_pil, mirror_type in mirror_list:
                        emb = self.get_embedding(img_pil)
                        emb_list.append(emb)
                        self.image_meta.append({
                            "original_name": img_name,
                            "mirror_type": mirror_type,
                            "image_path": os.path.abspath(img_path)
                        })
                else:
                    emb = self.get_embedding(img_path)
                    emb_list.append(emb)
                    self.image_meta.append({
                        "original_name": img_name,
                        "mirror_type": "original",
                        "image_path": os.path.abspath(img_path)
                    })

                if (idx + 1) % 20 == 0:
                    print(f"[构建索引] 已处理 {idx+1}/{len(img_paths)} 张原图，累计特征 {len(emb_list)} 条")
            except Exception as e:
                print(f"[构建索引] 跳过异常图片 {img_name}: {str(e)}")
                continue

        if not emb_list:
            raise RuntimeError("未成功提取任何有效特征，索引构建失败")

        emb_matrix = np.vstack(emb_list).astype(np.float32)
        emb_matrix = np.ascontiguousarray(emb_matrix)
        dim = emb_matrix.shape[1]
        
        print(f"[构建索引] 处理完成，共提取 {emb_matrix.shape[0]} 条特征，维度: {dim}")

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb_matrix)

        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.image_meta, f, ensure_ascii=False, indent=2)

        print(f"[构建索引] 完成！索引文件: {self.index_path}")
        print(f"[构建索引] 元数据文件: {self.meta_path}")

    def load_index(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError(
                "索引文件不存在，请先使用 --build 参数构建索引\n"
                f"预期索引路径: {self.index_path}"
            )

        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.image_meta = json.load(f)

        print(f"[加载索引] 成功加载 {self.index.ntotal} 条特征向量")
        print(f"[加载索引] 对应原始图片 {len(set(m['original_name'] for m in self.image_meta))} 张")

    def search(self, query_input, top_k: int = 3) -> list:
        if self.index is None:
            raise RuntimeError("索引未加载，请先调用 load_index() 或添加 --build 参数")

        query_emb = self.get_embedding(query_input)
        query_emb = np.expand_dims(query_emb, axis=0).astype(np.float32)

        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.image_meta):
                continue
            meta = self.image_meta[idx]
            results.append((
                float(score),      # 余弦相似度
                meta["original_name"],
                meta["image_path"],
                meta["mirror_type"],
            ))
        return results

    def _parse_drug_name(self, filename: str) -> str:
        """
        从文件名中提取药品名称
        文件名规范：医院id_设备id_药品名字_随机号.jpg
        兼容药品名称中包含下划线的场景
        """
        name_no_ext = os.path.splitext(filename)[0]
        parts = name_no_ext.split('_')
        # 前两段为医院ID、设备ID，最后一段为随机号，中间部分为药品名
        if len(parts) >= 3:
            return '_'.join(parts[2:-1])
        else:
            # 格式不符合时返回完整文件名（无后缀）作为兜底
            return name_no_ext

    def print_acc(self, eval_dir: str, top_k: int = 1) -> float:
        """
        遍历评估目录下的所有图片，计算真正的 Top-K 检索准确率
        判定标准：查询图片与检索出的 Top-K 结果中，只要有任意一个药品名称一致则计为正确
        """
        print(f"\n[准确率评估] 开始遍历评估目录: {eval_dir}")
        exts = (".jpg", ".jpeg", ".png", ".bmp")

        img_paths = []
        for root, _, files in os.walk(eval_dir):
            for file in files:
                if file.lower().endswith(exts):
                    img_paths.append(os.path.join(root, file))

        if not img_paths:
            raise ValueError(f"评估目录 {eval_dir} 及其子目录中未找到任何图片文件")

        correct = 0
        total = 0

        for idx, img_path in enumerate(img_paths):
            img_name = os.path.basename(img_path)
            try:
                query_drug = self._parse_drug_name(img_name)
                results = self.search(img_path, top_k=top_k)

                if not results:
                    total += 1
                    continue

                is_match = False
                for res in results:
                    result_filename = res[1]
                    result_drug = self._parse_drug_name(result_filename)
                    
                    if query_drug == result_drug:
                        is_match = True
                        break  # 只要命中一个，就停止检查当前图片剩余的结果

                if is_match:
                    correct += 1
                total += 1
                # ===============================================

                # 进度打印
                if (idx + 1) % 20 == 0:
                    current_acc = correct / total if total > 0 else 0.0
                    print(f"[准确率评估] 已处理 {idx+1}/{len(img_paths)} 张，当前准确率: {current_acc:.4f}")

            except Exception as e:
                print(f"[准确率评估] 跳过异常图片 {img_name}: {str(e)}")
                continue

        if total == 0:
            raise RuntimeError("未成功处理任何有效图片，无法计算准确率")

        accuracy = correct / total
        print("\n" + "=" * 70)
        print(f"[准确率评估] 评估完成")
        print(f"总测试样本数: {total}")
        print(f"正确匹配数: {correct}")
        print(f"Top-{top_k} 准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print("=" * 70)

        return accuracy


# ===================== 命令行入口 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于ONNX+Faiss的药盒图像检索工具（支持镜像增强）")
    parser.add_argument("--onnx_model", type=str, required=True, help="ONNX模型文件路径")
    parser.add_argument("--db_dir", type=str, required=True, help="药品图片数据库目录")
    parser.add_argument("--index_dir", type=str, default="./faiss_index", help="索引保存/加载目录")
    parser.add_argument("--query", type=str, default=None, help="待查询图片路径")
    parser.add_argument("--top_k", type=int, default=5, help="返回最相似的前N张结果")
    parser.add_argument("--build", action="store_true", help="执行前先重新构建索引")
    parser.add_argument("--no_mirror", action="store_true", help="构建索引时关闭镜像增强（推荐药盒识别时开启此参数）")
    parser.add_argument("--eval_dir", type=str, default=None, help="评估图片集目录，用于计算检索准确率（文件名需符合命名规范）")

    args = parser.parse_args()

    try:
        retriever = MedicineBoxONNXRetriever(
            onnx_model_path=args.onnx_model,
            db_dir=args.db_dir,
            index_dir=args.index_dir
        )

        # 构建索引
        if args.build:
            print("=" * 50)
            retriever.build_index(enable_mirror=not args.no_mirror)
            print("=" * 50)

        # 加载索引
        retriever.load_index()

        if args.eval_dir:
            retriever.print_acc(args.eval_dir, top_k=args.top_k)

        # 执行单图检索
        if args.query:
            print(f"\n[检索] 查询图片: {args.query}")
            results = retriever.search(args.query, top_k=args.top_k)

            # 格式化输出结果
            print("\n" + "=" * 70)
            print("检索结果（按相似度从高到低排序）")
            print("-" * 70)
            for i, res in enumerate(results, 1):
                print(f"排名 {i:2d} | 相似度: {res[0]:.4f} | 镜像类型: {res[3]}")
                print(f"       | 原图名: {res[1]}")
                print(f"       | 路径:   {res[2]}")
                print("-" * 70)

        # 参数校验：至少指定一个执行动作
        if not args.query and not args.eval_dir:
            print("\n⚠️  请至少传入 --query（单图检索）或 --eval_dir（批量评估）参数")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ 运行出错: {str(e)}")
        sys.exit(1)
