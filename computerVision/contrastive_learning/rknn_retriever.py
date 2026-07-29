# -*- coding:utf-8 -*-

import os
import sys
import argparse
import json
import pickle
import concurrent.futures
import numpy as np
import faiss
import queue
import time
import cv2
from PIL import Image

class MedicineBoxRetriever:
    def __init__(self, model_path, db_dir, index_dir="./faiss_index", img_size=320):
        self.db_dir = db_dir
        self.index_dir = index_dir
        # 移除未使用的 self.index_path
        self.meta_path = os.path.join(index_dir, "image_meta.json")
        self.input_size = (img_size, img_size)

        self.model_type = "rknn" if model_path.endswith(".rknn") else "onnx"
        self.rknn_pool = queue.Queue()
        
        # 温控与并发配置
        self.npu_cooling_delay = 0.002  # NPU 专属散热延迟(秒)

        if self.model_type == "onnx":
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
            self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"[模型加载] ONNX 模型加载成功，当前使用设备: {self.session.get_providers()[0]}")
            self.max_workers = max(4, os.cpu_count() or 4)

        elif self.model_type == "rknn":
            try:
                from rknnlite.api import RKNNLite
                self.is_lite = True
            except ImportError:
                print("\n❌ 错误: 未检测到 rknn-toolkit-lite2 环境。加速版并发需要运行在开发板上。")
                sys.exit(1)

            print(f"[模型加载] 正在加载 RKNN 模型: {model_path}")
            core_masks = [1, 2]
            for mask in core_masks:
                rknn = RKNNLite()
                ret = rknn.load_rknn(model_path)
                if ret != 0:
                    raise RuntimeError(f"RKNN 模型加载失败 (core_mask={mask})！")
                ret = rknn.init_runtime(core_mask=mask)
                if ret != 0:
                    raise RuntimeError(f"RKNN init_runtime 失败 (core_mask={mask})！")

                self.rknn_pool.put(rknn)
                print(f"[模型加载] RKNN 实例绑定成功 -> NPU Core Mask: {mask}")
            print(f"[模型加载] 2 核 NPU 均衡并发池初始化成功！")
            # 适配 RK3588 双核调度：2个工作线程对应 NPU，+1个用于缓冲 CPU 预处理
            self.max_workers = 3

        self.image_meta = []
        self.cluster_centers_index = None
        self.cluster_data = None
        self.all_cluster_embs = None  # 连续内存矩阵，极大提升反序列化速度
        self.cluster_names = []
        
        # 全局缓存文本特征池
        self.text_embs_cache = {}

    def _preprocess(self, img_pil):
        img_resized = img_pil.resize(self.input_size, Image.BILINEAR)
        img_np = np.array(img_resized, dtype=np.float32)

        if self.model_type == "onnx":
            img_np /= 255.0
            img_np = img_np.transpose((2, 0, 1))

        return np.expand_dims(img_np, axis=0)

    def _generate_mirror_images(self, img_path):
        try:
            img_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"图片文件损坏或无法读取: {img_path} | {e}")

        return [
            (img_pil, "original"),
            (img_pil.transpose(Image.FLIP_LEFT_RIGHT), "horizontal"),
            (img_pil.transpose(Image.FLIP_TOP_BOTTOM), "vertical"),
            (img_pil.transpose(Image.ROTATE_180), "diagonal")
        ]

    def get_embedding(self, image_input):
        if isinstance(image_input, str):
            img_pil = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img_pil = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
            else:
                img_pil = Image.fromarray(image_input).convert('RGB')
        else:
            raise TypeError("不支持的格式，请传入图片路径、PIL.Image 或 cv2图像帧")

        input_tensor = self._preprocess(img_pil)

        if self.model_type == "onnx":
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            embedding = outputs[0]
        else:
            rknn_instance = self.rknn_pool.get()
            outputs = rknn_instance.inference(inputs=[input_tensor], data_format='nhwc')
            # NPU 温控断点：在实例被下一个任务抢占前强制歇息，不阻塞当前 CPU 线程后续逻辑
            time.sleep(self.npu_cooling_delay)
            self.rknn_pool.put(rknn_instance)
            embedding = outputs[0]

        return embedding.squeeze().astype(np.float32)

    def _parse_drug_name(self, filename):
        name_no_ext = os.path.splitext(filename)[0]
        parts = name_no_ext.split('_')
        return parts[2][-7:] if len(parts) >= 3 else None

    def _fuse_embedding(self, img_emb, imag_name, alpha):
        if imag_name not in self.text_embs_cache:
            return img_emb

        text_emb = self.text_embs_cache[imag_name]
        
        if text_emb.shape[0] != img_emb.shape[0]:
            raise ValueError(f"图文特征维度不匹配！图像:{img_emb.shape[0]} vs 文本:{text_emb.shape[0]}")

        fused = img_emb * alpha + (1.0 - alpha) * text_emb
        norm = np.linalg.norm(fused)
        return fused / norm if norm > 1e-12 else fused

    def build_index(self, enable_mirror, text_embedding_dict=None, alpha=0.7):
        print(f"[构建索引] 开始遍历数据库目录: {self.db_dir}")
        print(f"[构建索引] 镜像增强: {'开启' if enable_mirror else '关闭'}")
        os.makedirs(self.index_dir, exist_ok=True)

        # 【优化】全局资源预先载入并转换为 numpy，消除每次遍历的开销
        self.text_embs_cache.clear()
        if text_embedding_dict is not None:
            print(f"[构建索引] 图文融合: 开启，图像权重 alpha={alpha}")
            for k, v in text_embedding_dict.items():
                self.text_embs_cache[k] = np.array(v, dtype=np.float32).squeeze()

        exts = (".jpg", ".jpeg", ".png", ".bmp")
        img_paths = [os.path.join(r, f) for r, _, fs in os.walk(self.db_dir) for f in fs if f.lower().endswith(exts)]

        if not img_paths:
            raise ValueError(f"目录 {self.db_dir} 中未找到任何图片")

        emb_list = []
        self.image_meta = []

        def _process_single_image(img_path):
            local_embs = []
            local_meta = []
            img_name = os.path.basename(img_path)
            imag_name = self._parse_drug_name(img_name)

            try:
                images_to_process = self._generate_mirror_images(img_path) if enable_mirror else [(Image.open(img_path).convert('RGB'), "original")]
                
                for img_pil, mirror_type in images_to_process:
                    emb = self.get_embedding(img_pil)
                    emb = self._fuse_embedding(emb, imag_name, alpha)
                    
                    local_embs.append(emb)
                    local_meta.append({
                        "original_name": img_name,
                        "mirror_type": mirror_type,
                        "image_path": os.path.abspath(img_path)
                    })
                return True, local_embs, local_meta
            except Exception as e:
                return False, img_name, str(e)

        print(f"[构建索引] 开启多线程特征提取池 (底层自适应温控)...")
        processed_count = 0
        total_count = len(img_paths)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_process_single_image, p): p for p in img_paths}

            for future in concurrent.futures.as_completed(futures):
                success, res1, res2 = future.result()
                if success:
                    emb_list.extend(res1)
                    self.image_meta.extend(res2)
                else:
                    print(f"[构建索引] 跳过异常图片 {res1}: {res2}")

                processed_count += 1
                if processed_count % 20 == 0:
                    print(f"[构建索引] 已处理 {processed_count}/{total_count} 张原图，累计特征 {len(emb_list)} 条")

        if not emb_list:
            raise RuntimeError("未成功提取任何有效特征，索引构建失败")

        emb_matrix = np.vstack(emb_list).astype(np.float32)
        emb_matrix = np.ascontiguousarray(emb_matrix)
        dim = emb_matrix.shape[1]

        print(f"[构建索引] 处理完成，共提取 {emb_matrix.shape[0]} 条特征，维度: {dim}")
        
        # 保存 meta 信息
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.image_meta, f, ensure_ascii=False, indent=2)

        print("[构建索引] 开始构建簇索引...")
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        emb_normalized = emb_matrix / norms

        cluster_to_indices = {}
        for idx, meta in enumerate(self.image_meta):
            drug_code = self._parse_drug_name(meta["original_name"])
            if drug_code:
                cluster_to_indices.setdefault(drug_code, []).append(idx)

        # 【优化】重构连续内存矩阵存储，绕过 pickle 反序列化碎片字典的灾难性性能
        cluster_names = []
        cluster_centers = []
        self.cluster_data = {}
        all_group_embs_list = []
        current_offset = 0

        for drug_code, indices in cluster_to_indices.items():
            group_embs = emb_normalized[indices]
            center = np.mean(group_embs, axis=0)
            center_norm = np.linalg.norm(center)
            if center_norm > 1e-12:
                center = center / center_norm
            
            cluster_centers.append(center)
            cluster_names.append(drug_code)
            all_group_embs_list.append(group_embs)
            
            size = len(indices)
            self.cluster_data[drug_code] = {
                "start": current_offset,
                "end": current_offset + size,
                "meta_indices": indices
            }
            current_offset += size

        if not cluster_centers:
            raise RuntimeError("没有有效的簇可构建索引，请检查图片命名规范")

        self.all_cluster_embs = np.vstack(all_group_embs_list).astype(np.float32)
        cluster_centers_matrix = np.vstack(cluster_centers).astype(np.float32)
        cdim = cluster_centers_matrix.shape[1]
        
        self.cluster_centers_index = faiss.IndexFlatIP(cdim)
        self.cluster_centers_index.add(cluster_centers_matrix)
        self.cluster_names = cluster_names

        faiss.write_index(self.cluster_centers_index, os.path.join(self.index_dir, "cluster_centers.index"))
        with open(os.path.join(self.index_dir, "cluster_data.pkl"), "wb") as f:
            # 只存三个连续块大对象，极速 Load
            pickle.dump({
                "cluster_names": self.cluster_names,
                "cluster_data": self.cluster_data,
                "all_embs": self.all_cluster_embs 
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[构建索引] 簇索引构建完成，共 {len(self.cluster_names)} 个簇")

    def load_index(self):
        # 移除了对废弃 HNSW 文件的检查
        cluster_centers_path = os.path.join(self.index_dir, "cluster_centers.index")
        cluster_data_path = os.path.join(self.index_dir, "cluster_data.pkl")

        if not os.path.exists(self.meta_path) or not os.path.exists(cluster_centers_path) or not os.path.exists(cluster_data_path):
            raise FileNotFoundError("索引文件不完整，请先使用 --build 参数重新构建索引")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.image_meta = json.load(f)
        
        self.cluster_centers_index = faiss.read_index(cluster_centers_path)
        with open(cluster_data_path, "rb") as f:
            data = pickle.load(f)
            self.cluster_names = data["cluster_names"]
            self.cluster_data = data["cluster_data"]
            self.all_cluster_embs = data["all_embs"]
            
        print(f"[加载索引] 成功加载簇索引，包含 {len(self.cluster_names)} 个簇, 共 {len(self.image_meta)} 条特征向量")

    def search(self, query_input, top_k: int = 3):
        if self.cluster_centers_index is None or self.cluster_data is None:
            raise RuntimeError("簇索引未加载，请重新构建索引")

        query_emb = self.get_embedding(query_input).astype(np.float32)
        norm = np.linalg.norm(query_emb)
        if norm > 1e-12:
            query_emb = query_emb / norm
        query_vec = query_emb.reshape(1, -1)

        scores_c, indices_c = self.cluster_centers_index.search(query_vec, top_k)
        results = []
        for score_c, idx_c in zip(scores_c[0], indices_c[0]):
            if idx_c < 0 or idx_c >= len(self.cluster_names):
                continue
                
            cluster_name = self.cluster_names[idx_c]
            info = self.cluster_data[cluster_name]
            
            # 【优化】通过 slice 切片内存连续块，替代字典拷贝
            embs = self.all_cluster_embs[info["start"]:info["end"]]
            meta_indices = info["meta_indices"]

            sims = np.dot(embs, query_vec.T).flatten()
            best_local_idx = int(np.argmax(sims))
            best_meta_idx = meta_indices[best_local_idx]
            meta = self.image_meta[best_meta_idx]

            results.append([
                float(sims[best_local_idx]),
                cluster_name,
                meta["image_path"],
                meta["mirror_type"]
            ])
        return results

    def print_acc(self, eval_dir, top_k=1):
        print(f"\n[准确率评估] 开始遍历评估目录: {eval_dir}")
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        img_paths = [os.path.join(r, f) for r, _, fs in os.walk(eval_dir) for f in fs if f.lower().endswith(exts)]

        if not img_paths:
            raise ValueError(f"评估目录 {eval_dir} 中未找到任何图片")

        def _eval_single(img_path):
            img_name = os.path.basename(img_path)
            try:
                query_drug = self._parse_drug_name(img_name)
                if not query_drug:
                    return False, False

                results = self.search(img_path, top_k=top_k)
                if not results:
                    return True, False

                for res in results:
                    if query_drug == res[1]:
                        return True, True
                return True, False
            except Exception as e:
                print(f"[准确率评估] 跳过异常图片 {img_name}: {str(e)}")
                return False, False

        correct, total = 0, 0
        # 评测阶段依旧遵循 NPU 自适应并发机制，拒绝资源干涉
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_eval_single, p) for p in img_paths]
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                success, is_match = future.result()
                if success:
                    total += 1
                    if is_match:
                        correct += 1

                if (idx + 1) % 20 == 0:
                    acc = correct / total if total > 0 else 0.0
                    print(f"[准确率评估] 已处理 {idx+1}/{len(img_paths)} 张，当前准确率: {acc:.4f}")

        if total == 0:
            raise RuntimeError("未成功处理任何有效图片，无法计算准确率")

        accuracy = correct / total
        print("\n" + "=" * 70)
        print(f"[准确率评估] 评估完成 | 样本: {total} | 正确: {correct} | Top-{top_k}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("=" * 70)
        return accuracy

    def __del__(self):
        if hasattr(self, 'model_type') and self.model_type == "rknn":
            while not self.rknn_pool.empty():
                rknn_instance = self.rknn_pool.get()
                rknn_instance.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于 RKNN/ONNX 的高并发层次化图像检索工具 (极致优化版)")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径 (支持 .onnx 或 .rknn)")
    parser.add_argument("--db_dir", type=str, required=True, help="药品图片数据库目录")
    parser.add_argument("--img_size", type=int, default=320, help="输入模型分辨率大小")
    parser.add_argument("--index_dir", type=str, default="./faiss_index", help="索引保存/加载目录")
    parser.add_argument("--query", type=str, default=None, help="待查询图片路径")
    parser.add_argument("--top_k", type=int, default=1, help="返回最相似的前N个簇结果")
    parser.add_argument("--build", action="store_true", help="执行前先重新构建索引")
    parser.add_argument("--no_mirror", action="store_true", help="构建索引时关闭镜像增强")
    parser.add_argument("--eval_dir", type=str, default=None, help="评估图片集目录")
    parser.add_argument("--text_emb_pkl", type=str, default=None, help="文本嵌入pkl文件路径")
    parser.add_argument("--alpha", type=float, default=0.7, help="图像特征权重，默认0.7")

    args = parser.parse_args()

    try:
        retriever = MedicineBoxRetriever(
            model_path=args.model,
            db_dir=args.db_dir,
            index_dir=args.index_dir,
            img_size=args.img_size
        )

        if args.build:
            print("=" * 50)
            text_embedding_dict = None
            if args.text_emb_pkl:
                with open(args.text_emb_pkl, 'rb') as f:
                    text_embedding_dict = pickle.load(f)
                print(f"[构建索引] 已加载文本字典，共 {len(text_embedding_dict)} 条文本特征")

            retriever.build_index(
                enable_mirror=not args.no_mirror,
                text_embedding_dict=text_embedding_dict,
                alpha=args.alpha
            )
            print("=" * 50)

        retriever.load_index()

        if args.eval_dir:
            retriever.print_acc(args.eval_dir, top_k=args.top_k)

        if args.query:
            results = retriever.search(args.query, top_k=args.top_k)
            print("\n" + "=" * 70)
            print("检索结果（按簇相似度从高到低排序）")
            print("-" * 70)
            for i, res in enumerate(results, 1):
                print(f"排名 {i:2d} | 相似度: {res[0]:.4f} | 簇名: {res[1]} | 镜像类型: {res[3]}\n       | 路径: {res[2]}\n" + "-" * 70)

        if not args.query and not args.eval_dir and not args.build:
            print("\n⚠️ 请传入 --query, --eval_dir 或 --build 参数")
            sys.exit(1)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ 运行出错: {str(e)}")
        sys.exit(1)
