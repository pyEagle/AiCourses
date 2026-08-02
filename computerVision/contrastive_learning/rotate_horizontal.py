import cv2
import numpy as np
import math
import os
import sys

class RotateHorizontal:
    def __init__(self, make_vertical=False, flip_180=False, 
                 bg_border_ratio=0.01, mask_kernel_ratio=0.01, 
                 sharpen_w1=1.3, sharpen_w2=-0.3):
        self.make_vertical = make_vertical
        self.flip_180 = flip_180
        
        self.bg_border_ratio = bg_border_ratio 
        self.mask_kernel_ratio = mask_kernel_ratio 
        self.sharpen_w1 = sharpen_w1
        self.sharpen_w2 = sharpen_w2

    def get_adaptive_kernel(self, w, h, ratio=0.01):
        base_size = min(w, h)
        k_size = max(3, int(base_size * ratio)) 
        k_size = k_size + 1 if k_size % 2 == 0 else k_size 
        return np.ones((k_size, k_size), np.uint8)

    def perspective_transform(self, orig_img, cropped_mask, crop_x, crop_y, M_affine):
        if orig_img is None or cropped_mask is None:
            return None
            
        contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest_contour, True)
        
        approx = None
        for eps in np.linspace(0.01, 0.1, 20):
            temp_approx = cv2.approxPolyDP(largest_contour, eps * peri, True)
            if len(temp_approx) == 4 and cv2.isContourConvex(temp_approx):
                approx = temp_approx
                break
                
        if approx is not None:
            pts = approx.reshape(4, 2)
        else:
            rect_minArea = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect_minArea)
            pts = np.array(box, dtype="float32")
            
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        sorted_pts = pts[np.argsort(angles)]
        
        s = sorted_pts.sum(axis=1)
        tl_idx = np.argmin(s)
        rect = np.roll(sorted_pts, shift=-tl_idx, axis=0).astype("float32")
        
        pts_rot = rect + np.array([crop_x, crop_y], dtype="float32")
        M_inv = cv2.invertAffineTransform(M_affine)
        pts_rot_expanded = np.array([pts_rot]) 
        pts_orig = cv2.transform(pts_rot_expanded, M_inv)[0]
        
        widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
        heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        if maxWidth <= 0 or maxHeight <= 0:
            return None
            
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
            
        M_persp = cv2.getPerspectiveTransform(pts_orig, dst)
        
        warped = cv2.warpPerspective(orig_img, M_persp, (maxWidth, maxHeight), 
                                     flags=cv2.INTER_LANCZOS4, 
                                     borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=(0, 0, 0, 0))

        dynamic_sigma = max(1.0, min(maxWidth, maxHeight) * 0.002)
        blurred = cv2.GaussianBlur(warped, (0, 0), dynamic_sigma)
        
        warped = cv2.addWeighted(warped, self.sharpen_w1, blurred, self.sharpen_w2, 0)
        
        return warped

    def pca_to_horizontal(self, img):
        if img is None:
            return None
            
        img = img.copy()
        
        if len(img.shape) == 3 and img.shape[2] == 3:
            h, w = img.shape[:2]
            
            border_size = max(1, int(min(h, w) * self.bg_border_ratio))
            
            top_border = img[0:border_size, :]
            bottom_border = img[h-border_size:h, :]
            left_border = img[:, 0:border_size]
            right_border = img[:, w-border_size:w]
            border_pixels = np.concatenate([
                top_border.reshape(-1, 3), 
                bottom_border.reshape(-1, 3), 
                left_border.reshape(-1, 3), 
                right_border.reshape(-1, 3)
            ], axis=0)
            bg_color = np.median(border_pixels, axis=0)
            
            diff = np.abs(img.astype(np.int32) - bg_color.astype(np.int32))
            max_diff = np.max(diff, axis=2).astype(np.uint8)
            
            _, alpha = cv2.threshold(max_diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = alpha

        if len(img.shape) < 3 or img.shape[2] != 4:
            print("跳过: 无法读取图像或没有Alpha通道")
            return None

        mask = img[:, :, 3] 
        
        h_orig, w_orig = img.shape[:2]
        
        kernel = self.get_adaptive_kernel(w_orig, h_orig, ratio=self.mask_kernel_ratio)
        
        eroded_mask = cv2.erode(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            clean_eroded_mask = np.zeros_like(mask)
            cv2.drawContours(clean_eroded_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            dilated_mask = cv2.dilate(clean_eroded_mask, kernel, iterations=2)
            mask = cv2.bitwise_and(mask, dilated_mask)
            img[:, :, 3] = mask

        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) == 0:
            print("跳过: 图像全透明或未识别到有效前景")
            return None
        
        coords = np.empty((len(x_coords), 2), dtype=np.float64)
        coords[:, 0] = x_coords
        coords[:, 1] = y_coords

        mean, eigenvectors, eigenvalues = cv2.PCACompute2(coords, np.empty((0)))
        dx, dy = eigenvectors[0][0], eigenvectors[0][1]
        
        angle = math.atan2(dy, dx) * 180.0 / math.pi
        rotation_angle = angle

        if self.make_vertical:
            rotation_angle += 90
        if self.flip_180:
            rotation_angle += 180

        center = (w_orig // 2, h_orig // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h_orig * sin) + (w_orig * cos))
        nH = int((h_orig * cos) + (w_orig * sin))

        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]

        pts = np.column_stack((x_coords, y_coords)).astype(np.float32)
        pts_expanded = np.array([pts]) 
        rotated_pts = cv2.transform(pts_expanded, M)[0]
        
        x, y, w, h = cv2.boundingRect(rotated_pts)

        if w <= 0 or h <= 0:
            print("跳过: 旋转后截取不到有效前景")
            return None
            
        clean_mask_temp = np.zeros((nH, nW), dtype=np.uint8)
        rotated_pts_int = np.int32(rotated_pts)
        rotated_pts_shifted = rotated_pts_int - np.array([x, y])
        
        cropped_mask = np.zeros((h, w), dtype=np.uint8)
        cropped_mask[rotated_pts_shifted[:, 1], rotated_pts_shifted[:, 0]] = 255
        
        cropped_mask = cv2.morphologyEx(cropped_mask, cv2.MORPH_CLOSE, kernel)

        cropped_bgra = self.perspective_transform(img, cropped_mask, x, y, M)

        if cropped_bgra is None:
            return None

        bgr = cropped_bgra[:, :, :3]
        if cropped_bgra.shape[2] == 4:
            alpha_mask = cropped_bgra[:, :, 3].astype(float) / 255.0
        else:
            alpha_mask = np.ones((cropped_bgra.shape[0], cropped_bgra.shape[1]), dtype=float)
        
        black_background = np.zeros_like(bgr, dtype=np.uint8)
        
        for c in range(3):
            black_background[:, :, c] = (bgr[:, :, c] * alpha_mask).astype(np.uint8)

        final_h, final_w = black_background.shape[:2]
        if final_h > final_w:
            black_background = cv2.rotate(black_background, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        return black_background

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
        image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                       if f.lower().endswith(valid_extensions)]
        
        if not image_paths:
            print(f"在目录 '{input_dir}' 中没有找到受支持的图像文件。")
            return

        print(f"找到 {len(image_paths)} 张图片，开始批量处理...\n" + "-"*40)
        
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

        result = cv2.bitwise_and(image_bgra, image_bgra, mask=binary_mask)

        return result

    def get_perfect_object(self, image, mask, output_path="debug_perfect_output.png", debug=True):
        real_object_img = self.get_real_object(image, mask)
        
        if real_object_img is None:
            print("无法提取 real_object，处理中止。")
            return None
            
        perfect_object_img = self.buffer_proc_image(real_object_img, output_path=output_path, debug=debug)
        
        return perfect_object_img


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python script.py <输入文件夹路径> <输出文件夹路径>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    processor = RotateHorizontal(make_vertical=False, flip_180=False)
    processor.batch_proc_image(input_directory, output_directory)

