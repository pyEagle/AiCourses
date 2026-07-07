import os
import cv2
import argparse
import numpy as np
import onnxruntime as ort


def get_color(label):
    if label == "s":  # s  - 绿色
        return (0, 255, 0)
    elif label == "y":  # y  - 蓝色
        return (255, 0, 0)

    return None


class YOLOv8MaskDetector:
    CLASSES = ["2c", "3c", "4c", "5c", "s", "y"]  # nc=6，已删除 sy
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    # COLORS = [
    #    (0, 255, 0),    # s  - 绿色
    #    (0, 255, 255),  # sy - 黄色
    #    (255, 0, 0),    # y  - 蓝色
    #    ]
    OBJ_THRESH = 0.25
    NMS_THRESH = 0.70
    IMG_SIZE = (320, 320)

    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None
        self.input_name = None

        self._load_model()

    def _load_model(self):
        print(f"[Info] 正在加载 ONNX 模型: {self.model_path}")
        providers = self._get_optimal_providers()
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print("[Info] 模型加载完成")

    def _get_optimal_providers(self):
        available_providers = ort.get_available_providers()
        print(f"[Info] 已自动检测并启用的硬件加速优先级列表: {available_providers}")
        return available_providers

    def _letterbox(self, img, new_shape=IMG_SIZE, color=(114, 114, 114)):
        shape = img.shape[:2]  # current shape [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return img, (r, r), (dw, dh)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (
                (img1_shape[1] - img0_shape[1] * gain) / 2,
                (img1_shape[0] - img0_shape[0] * gain) / 2,
            )
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        boxes[:, [0, 2]] -= pad[0]  # x padding
        boxes[:, [1, 3]] -= pad[1]  # y padding
        boxes[:, :4] /= gain

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0])
        return boxes

    def _crop_mask(self, masks, boxes):
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
        c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def _process_mask(self, protos, masks_in, bboxes, shape, upsampled_size):
        c, mh, mw = protos.shape  # CHW (32, 160, 160)
        ih, iw = shape

        masks = self._sigmoid(masks_in @ protos.reshape(c, -1)).reshape(
            -1, mh, mw
        )  # (n, 160, 160)

        downsampled_bboxes = bboxes.copy()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 1] *= mh / ih
        downsampled_bboxes[:, 3] *= mh / ih

        masks = self._crop_mask(masks, downsampled_bboxes)

        out_masks = np.zeros(
            (masks.shape[0], upsampled_size[0], upsampled_size[1]), dtype=np.float32
        )
        for i in range(masks.shape[0]):
            out_masks[i] = cv2.resize(
                masks[i],
                (upsampled_size[1], upsampled_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        return out_masks > 0.5

    def _postprocess(self, preds, img_ori_shape, img_pad_shape, ratio_pad):
        preds_det = preds[0][0]  # shape (4+6+32, 8400) -> 42: (box+cls+mask_coeffs)
        protos = preds[1][0]  # shape (32, 160, 160)

        preds_det = preds_det.T  # (8400, 43)
        boxes = preds_det[:, :4]
        scores = preds_det[:, 4 : 4 + len(self.CLASSES)]
        mask_coeffs = preds_det[:, 4 + len(self.CLASSES) :]

        boxes_xyxy = np.empty_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        boxes_nms = np.empty_like(boxes)
        boxes_nms[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x_min
        boxes_nms[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y_min
        boxes_nms[:, 2] = boxes[:, 2]  # width
        boxes_nms[:, 3] = boxes[:, 3]  # height

        class_ids = np.argmax(scores, axis=1)
        class_scores = np.max(scores, axis=1)

        mask = class_scores > self.OBJ_THRESH
        boxes_xyxy = boxes_xyxy[mask]
        boxes_nms = boxes_nms[mask]
        class_scores = class_scores[mask]
        class_ids = class_ids[mask]
        mask_coeffs = mask_coeffs[mask]

        if len(class_scores) == 0:
            return [], [], [], []

        max_wh = 7680
        boxes_nms[:, 0] += class_ids * max_wh
        boxes_nms[:, 1] += class_ids * max_wh

        indices = cv2.dnn.NMSBoxes(
            boxes_nms.tolist(), class_scores.tolist(), self.OBJ_THRESH, self.NMS_THRESH
        )
        if len(indices) == 0:
            return [], [], [], []

        indices = indices.flatten()

        boxes_xyxy = boxes_xyxy[indices]
        class_scores = class_scores[indices]
        class_ids = class_ids[indices]
        mask_coeffs = mask_coeffs[indices]

        masks = self._process_mask(
            protos, mask_coeffs, boxes_xyxy, img_pad_shape, img_pad_shape
        )

        boxes_ori = self._scale_boxes(
            img_pad_shape, boxes_xyxy, img_ori_shape, ratio_pad
        )

        pad_w, pad_h = int(ratio_pad[1][0]), int(ratio_pad[1][1])
        masks_ori = masks[
            :, pad_h : img_pad_shape[0] - pad_h, pad_w : img_pad_shape[1] - pad_w
        ]

        final_masks = np.zeros(
            (masks_ori.shape[0], img_ori_shape[0], img_ori_shape[1]), dtype=bool
        )
        for i in range(masks_ori.shape[0]):
            final_masks[i] = (
                cv2.resize(
                    masks_ori[i].astype(np.uint8),
                    (img_ori_shape[1], img_ori_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0.5
            )

        return boxes_ori, class_ids, class_scores, final_masks

    def _get_status(self, y_mask, c_box):
        """判断 y 的 mask 质心是否在 c 的 bounding box 内。
        在内部 → '1'，在外部或无检测 → '0'
        """
        if c_box is None:
            return "0"
        y_indices = np.argwhere(y_mask)  # [[row, col], ...]
        if len(y_indices) == 0:
            return "0"
        cy, cx = y_indices.mean(axis=0).astype(int)  # 质心 (row→y, col→x)
        x1, y1, x2, y2 = c_box.astype(int)
        return "1" if (x1 <= cx <= x2 and y1 <= cy <= y2) else "0"

    def _draw_result(
        self,
        img,
        boxes,
        classes,
        scores,
        masks,
    ):
        img_draw = img.copy()

        for i in range(len(boxes)):
            box = boxes[i].astype(int)
            cls_id = int(classes[i])
            score = scores[i]
            mask = masks[i]
            color = get_color(self.CLASSES[cls_id])
            if not color:
                continue
            color = self.COLORS[cls_id]
            # if not color: continue

            # 画出 Mask 轮廓边界
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(img_draw, contours, -1, color, 2)

            # 画 Box
            cv2.rectangle(img_draw, (box[0], box[1]), (box[2], box[3]), color, 2)

            # 画 Label
            label = f"{self.CLASSES[cls_id]} {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(
                img_draw, (box[0], box[1] - 20), (box[0] + w, box[1]), color, -1
            )
            cv2.putText(
                img_draw,
                label,
                (box[0], box[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            # 画 Box
            # cv2.rectangle(img_draw, (box[0], box[1]), (box[2], box[3]), color, 2)

            # 画 Label
            # (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            # cv2.rectangle(img_draw, (box[0], box[1] - 20), (box[0] + w, box[1]), color, -1)
            # #判断状态
            # status = self._get_status(mask, c_mask)
            # cv2.putText(img_draw, f"Status: {status}", (box[0], box[1] - 30),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if status == "1" else (0, 0, 255), 1)

        return img_draw

    def run(self, img_path_or_array):
        if isinstance(img_path_or_array, str):
            img_src = cv2.imread(img_path_or_array)
            if img_src is None:
                raise ValueError(f"[Error] 读取图像失败: {img_path_or_array}")
        elif isinstance(img_path_or_array, np.ndarray):
            img_src = img_path_or_array.copy()
        else:
            raise ValueError("img_path_or_array 必须是图像路径字符串或numpy数组")

        img_ori_shape = img_src.shape[:2]

        img_pad, ratio, pad = self._letterbox(img_src, self.IMG_SIZE)
        img_input = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)
        img_input = (
            img_input.transpose((2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
            / 255.0
        )

        outputs = self.session.run(None, {self.input_name: img_input})

        ratio_pad = (ratio, pad)
        boxes, class_ids, scores, masks = self._postprocess(
            outputs, img_ori_shape, img_pad.shape[:2], ratio_pad
        )

        vis_img = img_src.copy()
        if len(boxes) > 0:
            vis_img = self._draw_result(img_src, boxes, class_ids, scores, masks)
        else:
            print("[Info] 未检测到任何目标。")

        return boxes, class_ids, scores, masks, vis_img

    def save_result(self, vis_img, save_path):
        cv2.imwrite(save_path, vis_img)
        print(f"[Info] 结果已保存至 {save_path}")


def draw_overlay(vis_img, draw_info):
    """
    在 vis_img 上叠加 y 类的质心圆点和 status 文字。
    draw_info: list of (cx, cy, status)  — 由 video106.py 传入
    """
    COLOR_CENTER = (0, 0, 255)  # 红色质心圆点
    COLOR_STATUS = (0, 255, 255)  # 黄色 status 文字
    for cx, cy, status in draw_info:
        cv2.circle(vis_img, (cx, cy), radius=5, color=COLOR_CENTER, thickness=-1)
        cv2.putText(
            vis_img,
            f"status:{status}",
            (cx - 20, cy - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_STATUS,
            2,
            cv2.LINE_AA,
        )
    return vis_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="ONNX 模型路径")
    parser.add_argument("--img_path", type=str, required=True, help="测试图像路径")
    parser.add_argument("--out_dir", type=str, default="./result", help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 初始化检测器
    detector = YOLOv8MaskDetector(args.model_path)

    # 2. 执行推理
    boxes, class_ids, scores, masks, vis_img = detector.run(args.img_path)

    # 3. 保存结果
    if len(boxes) > 0:
        save_name = os.path.basename(args.img_path)
        save_path = os.path.join(args.out_dir, save_name)
        detector.save_result(vis_img, save_path)

    # 4. 返回结果（可选）
    print(f"[Info] 检测到 {len(boxes)} 个目标")
    for i in range(len(boxes)):
        print(
            f"  目标 {i+1}: 类别={detector.CLASSES[class_ids[i]]}, 置信度={scores[i]:.3f}, 框={boxes[i].astype(int)}"
        )
      
