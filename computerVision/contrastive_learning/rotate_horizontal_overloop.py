import cv2
import numpy as np
import math
import os
import sys

class RotateHorizontal:
    def __init__(self, make_vertical=False, flip_180=False):
        self.make_vertical = make_vertical
        self.flip_180 = flip_180

    def pca_to_horizontal(self, img):
        if img is None:
            return None
            
        if len(img.shape) == 3 and img.shape[2] == 3:
            h, w = img.shape[:2]
            corners = np.array([img[0, 0], img[0, w-1], img[h-1, 0], img[h-1, w-1]])
            bg_color = np.median(corners, axis=0)
            
            diff = np.abs(img.astype(np.int16) - bg_color.astype(np.int16))
            max_diff = np.max(diff, axis=2)
            
            alpha = ((max_diff > 15) * 255).astype(np.uint8)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = alpha

        if len(img.shape) < 3 or img.shape[2] != 4:
            print("跳过: 无法读取图像或没有Alpha通道")
            return None

        alpha_channel = img[:, :, 3]
        _, mask = cv2.threshold(alpha_channel, 20, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((7, 7), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            clean_eroded_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(clean_eroded_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            dilated_mask = cv2.dilate(clean_eroded_mask, kernel, iterations=2)
            mask = cv2.bitwise_and(mask, dilated_mask)
            
            img[:, :, 3] = mask
            
        x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(mask)
        if w_bbox == 0 or h_bbox == 0:
            print("跳过: 图像全透明或未识别到有效前景")
            return None
            
        img = img[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox]
        mask = mask[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox]

        y_coords, x_coords = np.where(mask > 0)
        
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

        (h_img, w_img) = img.shape[:2]
        center = (w_img // 2, h_img // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h_img * sin) + (w_img * cos))
        nH = int((h_img * cos) + (w_img * sin))

        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]

        rotated_img = cv2.warpAffine(img, M, (nW, nH), 
                                     borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=(0, 0, 0, 0))

        a = rotated_img[:, :, 3]
        _, a_thresh = cv2.threshold(a, 20, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(a_thresh)
        
        cropped_bgra = rotated_img[y:y+h, x:x+w]
        bgr = cropped_bgra[:, :, :3]
        
        alpha_mask = cropped_bgra[:, :, 3:4] / 255.0  # 切片保持 3D 维度 (h, w, 1)
        black_background = (bgr * alpha_mask).astype(np.uint8)
            
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

        x, y, w, h = cv2.boundingRect(binary_mask)
        
        if w == 0 or h == 0:
            print("警告: mask中未找到有效前景区域。")
            return None
            
        cropped_image = image[y:y+h, x:x+w]
        cropped_mask = binary_mask[y:y+h, x:x+w]

        image_bgra = cropped_image
        if len(cropped_image.shape) == 2:
            image_bgra = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGRA)
        elif len(cropped_image.shape) == 3 and cropped_image.shape[2] == 3:
            image_bgra = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)

        result = cv2.bitwise_and(image_bgra, image_bgra, mask=cropped_mask)

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
