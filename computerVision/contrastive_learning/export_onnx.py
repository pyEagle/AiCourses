import os
import argparse
import torch
import torch.nn as nn
from ultralytics import YOLO

# ==========================================
# 导出专用的特征提取网络 (去除了 F.normalize)
# ==========================================
class YOLOv8NeckExtractor(nn.Module):
    def __init__(self, model_file):
        super().__init__()
        yolo = YOLO(model_file)
        
        self.core_model = yolo.model 
        self.backbone_neck = self.core_model.model 
        
        for i, module in enumerate(self.backbone_neck):
            if i < 5: 
                for param in module.parameters():
                    param.requires_grad = False
        
        # 🌟 优化：彻底移除 Hook 机制相关的变量
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.embedding_head = nn.Sequential(
            nn.Flatten(1),
            # 保持 LazyLinear，由外部 dummy_input 推断形状
            nn.LazyLinear(512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256) 
        )

    def forward(self, x):
        # 🌟 核心优化：不执行整个模型，避免追踪无用的 YOLO 检测头
        # YOLOv8 的 0-9 层（直至 SPPF）是严格顺序的单支路，直接循环推理即可
        for i in range(10): 
            x = self.backbone_neck[i](x)
            
        # 此时的 x 就是原先 hook 获取到的 features（第 9 层输出）
        avg_f = self.avg_pool(x)
        max_f = self.max_pool(x)
        base_emb = torch.cat([avg_f, max_f], dim=1) 
        
        final_emb = self.embedding_head(base_emb)
        # ⚠️ 注意：这里故意去除了 F.normalize(final_emb, p=2, dim=1)
        # 对应导出文件名为 no_norm，需要在推理代码侧使用 numpy 手动计算归一化
        return final_emb

def export_custom_model(pth_path, base_yolo_pt, target_size=320, opset=12):
    print(f"📦 1. 正在初始化基础网络骨架: {base_yolo_pt}")
    try:
        model = YOLOv8NeckExtractor(base_yolo_pt).to('cpu')
    except Exception as e:
        print(f"❌ 初始化基础网络失败: {e}")
        return

    print(f"📥 2. 正在加载微调后的权重: {pth_path}")
    try:
        model.eval()
        # 🌟 提前进行一次前向传播，让 LazyLinear 自动推断形状并初始化权重张量
        dummy_init = torch.randn(1, 3, target_size, target_size, device='cpu')
        with torch.no_grad():
            _ = model(dummy_init)
        
        # 现在加载状态字典就不会有键值匹配错误了
        state_dict = torch.load(pth_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        model.eval()
    except Exception as e:
        print(f"❌ 加载自定义权重失败: {e}")
        return

    print(f"🚀 3. 正在生成计算图并导出 ONNX (尺寸: {target_size}x{target_size})...")
    dummy_input = torch.randn(1, 3, target_size, target_size, device='cpu')

    dir_name = os.path.dirname(pth_path)
    base_name = os.path.basename(pth_path)
    name, _ = os.path.splitext(base_name)
    output_onnx_path = os.path.join(dir_name, f"{name}_{target_size}_no_norm.onnx")

    try:
        torch.onnx.export(
            model,                        
            dummy_input,                  
            output_onnx_path,             
            export_params=True,           
            opset_version=opset,          
            do_constant_folding=True,     
            input_names=['images'],       
            output_names=['embeddings'],  
            dynamic_axes={'images': {0: 'batch_size'}, 'embeddings': {0: 'batch_size'}}
        )
        print(f"\n✅ ONNX 基础导出成功！")
        print("ℹ️  已移除模型内 L2 归一化层，需在推理后处理阶段手动补全归一化")

        # 尝试使用 onnxsim 简化模型
        try:
            import onnx
            from onnxsim import simplify
            print("⏳ 正在使用 onnxsim 简化模型...")
            onnx_model = onnx.load(output_onnx_path)
            model_simp, check = simplify(onnx_model)
            if check:
                onnx.save(model_simp, output_onnx_path)
                print("✨ 模型简化成功！图结构已达最优。")
            else:
                print("⚠️ 模型简化验证失败，已保留原版 ONNX。")
        except ImportError:
            print("💡 提示：未安装 onnxsim，跳过图简化步骤。推荐使用 `pip install onnxsim` 获取更好性能。")
        except Exception as e:
            print(f"⚠️ 简化步骤遇到问题跳过: {e}")

        print(f"\n🎯 最终模型已保存至: {output_onnx_path}")
        print(f"--> 输入节点: 'images' (动态形状: [batch_size, 3, {target_size}, {target_size}])")
        print(f"--> 输出节点: 'embeddings' (动态形状: [batch_size, 256])")
        
    except Exception as e:
        print(f"\n❌ ONNX 导出失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="药盒特征提取器 ONNX 导出工具（与训练脚本结构对齐）")
    
    parser.add_argument("--weights", type=str, required=True, help="微调好的 .pth 文件路径")
    parser.add_argument("--base", type=str, default="best_n_7.1.pt", help="基础 YOLOv8 .pt 模型路径")
    parser.add_argument("--size", type=int, default=320, help="输入图片尺寸 (默认: 320)")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset (默认: 12)")
    
    args = parser.parse_args()
    
    export_custom_model(
        pth_path=args.weights,
        base_yolo_pt=args.base,
        target_size=args.size,
        opset=args.opset
    )
