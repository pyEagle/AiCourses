import cv2
import numpy as np
import math
import os
import sys


class RotateHorizontal:
    def __init__(self, make_vertical=False, flip_180=False, padding=5):
        self.make_vertical = make_vertical
        self.flip_180 = flip_180
        self.padding = padding  # 裁剪后四周 uniform padding（像素）

    def pca_to_horizontal(self, img):
        if img is None:
            return None

        img = img.copy()

        # BGR → BGRA：用边缘像素估算背景色生成 alpha
        if len(img.shape) == 3 and img.shape[2] == 3:
            h, w = img.shape[:2]
            edge_pixels = np.concatenate([
                img[0, :], img[h - 1, :], img[:, 0], img[:, w - 1]
            ], axis=0)
            bg_color = np.median(edge_pixels, axis=0)

            diff = np.abs(img.astype(np.int32) - bg_color.astype(np.int32))
            max_diff = np.max(diff, axis=2)
            alpha = np.where(max_diff > 8, 255, 0).astype(np.uint8)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = alpha

        if len(img.shape) < 3 or img.shape[2] != 4:
            print("跳过: 无法读取图像或没有Alpha通道")
            return None

        alpha_channel = img[:, :, 3]
        _, mask = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)

        # 闭运算连接碎片，保留所有有效轮廓
        close_kernel = np.ones((21, 21), np.uint8)
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            img_area = mask.shape[0] * mask.shape[1]
            min_area = img_area * 0.0005
            clean_mask = np.zeros_like(mask)
            for contour in contours:
                if cv2.contourArea(contour) >= min_area:
                    cv2.drawContours(clean_mask, [contour], -1, 255, thickness=cv2.FILLED)
            mask = cv2.bitwise_and(mask, clean_mask)
            img[:, :, 3] = mask

        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) == 0:
            print("跳过: 图像全透明或未识别到有效前景")
            return None

        # 降采样加速
        max_points = 50000
        if len(x_coords) > max_points:
            indices = np.random.RandomState(42).choice(len(x_coords), max_points, replace=False)
            x_s, y_s = x_coords[indices], y_coords[indices]
        else:
            x_s, y_s = x_coords, y_coords

        # minAreaRect 计算旋转角度
        points = np.column_stack((x_s, y_s)).astype(np.int32)
        rect = cv2.minAreaRect(points)
        w_rect, h_rect = rect[1]
        angle_rect = rect[2]

        rotation_angle = angle_rect if w_rect >= h_rect else angle_rect + 90
        if rotation_angle > 90:
            rotation_angle -= 180
        elif rotation_angle < -90:
            rotation_angle += 180

        if self.make_vertical:
            rotation_angle += 90
        if self.flip_180:
            rotation_angle += 180

        # 旋转
        h_img, w_img = img.shape[:2]
        center = (w_img // 2, h_img // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

        cos_v = np.abs(M[0, 0])
        sin_v = np.abs(M[0, 1])
        nW = int(h_img * sin_v + w_img * cos_v)
        nH = int(h_img * cos_v + w_img * sin_v)
        M[0, 2] += nW / 2 - center[0]
        M[1, 2] += nH / 2 - center[1]

        rotated_img = cv2.warpAffine(
            img, M, (nW, nH),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # 【核心修改】紧裁 + 均匀 padding，替代原来的对称扩展逻辑
        a = rotated_img[:, :, 3]
        _, a_thresh = cv2.threshold(a, 10, 255, cv2.THRESH_BINARY)

        # 找前景精确边界
        col_has = np.any(a_thresh > 0, axis=0)
        row_has = np.any(a_thresh > 0, axis=1)

        if not np.any(col_has) or not np.any(row_has):
            return None

        col_idx = np.where(col_has)[0]
        row_idx = np.where(row_has)[0]

        left = col_idx[0]
        right = col_idx[-1]
        top = row_idx[0]
        bottom = row_idx[-1]

        # 紧裁边界 + uniform padding
        pad = self.padding
        crop_left = max(0, left - pad)
        crop_top = max(0, top - pad)
        crop_right = min(nW - 1, right + pad)
        crop_bottom = min(nH - 1, bottom + pad)

        cropped_bgra = rotated_img[crop_top:crop_bottom + 1, crop_left:crop_right + 1]

        return cropped_bgra

    def save_sub_image(self, img_matrix, output_path):
        if img_matrix is not None:
            out_dir = os.path.dirname(output_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)
            cv2.imwrite(output_path, img_matrix)
            print(f"保存成功: '{output_path}'")
        else:
            print(f"保存失败: 图像矩阵为空，无法保存至 '{output_path}'")

    def one_proc_image(self, input_path, output_path):
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"读取失败或文件不存在: '{input_path}'")
            return
        rotated_img = self.pca_to_horizontal(img)
        self.save_sub_image(rotated_img, output_path)

    def batch_proc_image(self, input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        image_paths = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith(valid_extensions)
        ]

        if not image_paths:
            print(f"在目录 '{input_dir}' 中没有找到受支持的图像文件。")
            return

        print(f"找到 {len(image_paths)} 张图片，开始批量处理...\n" + "-" * 40)

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            out_path = os.path.join(output_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                rotated_img = self.pca_to_horizontal(img)
                self.save_sub_image(rotated_img, out_path)
            else:
                print(f"读取失败: '{img_path}'")

        print("-" * 40 + "\n全部处理完成！")

    def buffer_proc_image(self, img_matrix, output_path="debug_output.png", debug=True):
        rotated_img = self.pca_to_horizontal(img_matrix)
        if debug and rotated_img is not None:
            self.save_sub_image(rotated_img, output_path)
        return rotated_img

    def get_real_object(self, image, mask):
        if image is None or mask is None:
            print("错误: image 或 mask 输入为空。")
            return None

        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        if len(image.shape) == 2:
            image_bgra = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        else:
            image_bgra = image.copy()

        return cv2.bitwise_and(image_bgra, image_bgra, mask=binary_mask)

    def get_perfect_object(self, image, mask, output_path="debug_perfect_output.png", debug=True):
        real_object_img = self.get_real_object(image, mask)
        if real_object_img is None:
            print("无法提取 real_object，处理中止。")
            return None
        return self.buffer_proc_image(real_object_img, output_path=output_path, debug=debug)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python script.py <输入文件夹路径> <输出文件夹路径>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    processor = RotateHorizontal(make_vertical=False, flip_180=False, padding=5)
    processor.batch_proc_image(input_directory, output_directory)
