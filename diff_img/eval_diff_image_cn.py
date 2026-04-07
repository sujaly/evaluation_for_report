import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class XRayImageComparatorCN:
    def __init__(self, image1_path, image2_path):
        self.image1_path = image1_path
        self.image2_path = image2_path
        self.image1 = None
        self.image2 = None
        self.diff_image = None
        self.lesion_boxes = []
        
    def load_images(self):
        self.image1 = cv2.imread(self.image1_path)
        self.image2 = cv2.imread(self.image2_path)
        
        if self.image1 is None or self.image2 is None:
            raise ValueError("无法加载图片，请检查文件路径")
        
        print(f"图片1尺寸: {self.image1.shape}")
        print(f"图片2尺寸: {self.image2.shape}")
        
    def align_images(self):
        if self.image1.shape != self.image2.shape:
            print("图片尺寸不同，进行对齐...")
            min_height = min(self.image1.shape[0], self.image2.shape[0])
            min_width = min(self.image1.shape[1], self.image2.shape[1])
            
            self.image1 = cv2.resize(self.image1, (min_width, min_height))
            self.image2 = cv2.resize(self.image2, (min_width, min_height))
            
            print(f"对齐后尺寸: {self.image1.shape}")
    
    def compute_difference(self):
        gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray1, gray2)
        
        self.diff_image = diff
        
        return diff
    
    def detect_lesions(self, threshold=30, min_area=500, max_lesions=8, merge_distance=100):
        if self.diff_image is None:
            self.compute_difference()
        
        _, binary = cv2.threshold(self.diff_image, threshold, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                raw_boxes.append({
                    'box': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area,
                    'intensity': np.mean(self.diff_image[y:y+h, x:x+w])
                })
        
        raw_boxes.sort(key=lambda x: x['area'], reverse=True)
        
        merged_boxes = []
        used = set()
        
        for i, box in enumerate(raw_boxes):
            if i in used:
                continue
            
            merged_box = box.copy()
            merged_indices = [i]
            
            for j in range(i + 1, len(raw_boxes)):
                if j in used:
                    continue
                
                other_box = raw_boxes[j]
                dist = np.sqrt((box['center'][0] - other_box['center'][0]) ** 2 + 
                              (box['center'][1] - other_box['center'][1]) ** 2)
                
                if dist < merge_distance:
                    x1 = min(merged_box['box'][0], other_box['box'][0])
                    y1 = min(merged_box['box'][1], other_box['box'][1])
                    x2 = max(merged_box['box'][0] + merged_box['box'][2], other_box['box'][0] + other_box['box'][2])
                    y2 = max(merged_box['box'][1] + merged_box['box'][3], other_box['box'][1] + other_box['box'][3])
                    
                    merged_box['box'] = (x1, y1, x2 - x1, y2 - y1)
                    merged_box['center'] = ((x1 + x2) // 2, (y1 + y2) // 2)
                    merged_box['area'] += other_box['area']
                    merged_indices.append(j)
                    used.add(j)
            
            merged_boxes.append(merged_box)
        
        self.lesion_boxes = merged_boxes[:max_lesions]
        
        for i, box in enumerate(self.lesion_boxes):
            box['id'] = i
            box['color'] = self.get_color_by_id(i)
        
        print(f"检测到 {len(self.lesion_boxes)} 个主要病灶区域 (原始: {len(raw_boxes)} 个)")
        return self.lesion_boxes
    
    def get_color_by_id(self, idx):
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 0, 255),
        ]
        return colors[idx % len(colors)]
    
    def analyze_progression(self):
        if not self.lesion_boxes:
            return "未检测到明显病灶变化"
        
        total_area = sum(box['area'] for box in self.lesion_boxes)
        avg_intensity = np.mean([box['intensity'] for box in self.lesion_boxes])
        
        if total_area > 10000 or avg_intensity > 80:
            progression = "病情进展明显"
            severity = "严重"
        elif total_area > 5000 or avg_intensity > 60:
            progression = "病情有进展"
            severity = "中等"
        elif total_area > 1000 or avg_intensity > 40:
            progression = "病情轻微变化"
            severity = "轻微"
        else:
            progression = "病情稳定"
            severity = "稳定"
        
        return {
            'progression': progression,
            'severity': severity,
            'total_area': total_area,
            'avg_intensity': avg_intensity,
            'lesion_count': len(self.lesion_boxes)
        }
    
    def draw_lesion_boxes(self, image, thickness=4):
        result = image.copy()
        
        for box_info in self.lesion_boxes:
            x, y, w, h = box_info['box']
            color = box_info.get('color', (0, 255, 0))
            lesion_id = box_info.get('id', 0)
            
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            
            label = f"B{lesion_id + 1}"
            cv2.putText(result, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)
        
        return result
    
    def visualize_comparison(self, save_path=None):
        if self.image1 is None or self.image2 is None:
            self.load_images()
            self.align_images()
        
        if self.diff_image is None:
            self.compute_difference()
        
        if not self.lesion_boxes:
            self.detect_lesions()
        
        progression_info = self.analyze_progression()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        img1_rgb = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
        
        img1_with_boxes = self.draw_lesion_boxes(self.image1)
        img1_with_boxes_rgb = cv2.cvtColor(img1_with_boxes, cv2.COLOR_BGR2RGB)
        
        img2_with_boxes = self.draw_lesion_boxes(self.image2)
        img2_with_boxes_rgb = cv2.cvtColor(img2_with_boxes, cv2.COLOR_BGR2RGB)
        
        axes[0, 0].imshow(img1_rgb)
        axes[0, 0].set_title('图像1 (早期)', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img2_rgb)
        axes[0, 1].set_title('图像2 (晚期)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(img1_with_boxes_rgb)
        axes[1, 0].set_title('图像1 - 病灶标记', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(img2_with_boxes_rgb)
        axes[1, 1].set_title('图像2 - 病灶标记', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        info_text = f"""
        病情进展报告:
        ========================
        进展评估: {progression_info['progression']}
        严重程度: {progression_info['severity']}
        病灶数量: {progression_info['lesion_count']}
        总面积: {progression_info['total_area']:.0f} 像素²
        平均强度: {progression_info['avg_intensity']:.1f}
        """
        
        plt.figtext(0.5, -0.05, info_text, ha='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results saved to: {save_path}")
        
        plt.close()
        
        return progression_info
    
    def create_difference_heatmap(self, save_path=None):
        if self.diff_image is None:
            self.compute_difference()
        
        heatmap = cv2.applyColorMap(self.diff_image, cv2.COLORMAP_JET)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        img1_rgb = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        axes[0].imshow(img1_rgb)
        axes[0].set_title('图像1 (早期)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(img2_rgb)
        axes[1].set_title('图像2 (当前)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(heatmap_rgb)
        axes[2].set_title('差异热力图 (红色=高差异)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")
        
        plt.close()

def main():
    image1_path = r"D:\pyworkspace\evaluation_for_report\data\jpg\p11000566\s50230446\2e4678a5-e646a648-d6265814-63b082e3-d14f047a.jpg"
    image2_path = r"D:\pyworkspace\evaluation_for_report\data\jpg\p11000566\s50252971\99eb5ea2-76aff341-b0db7fe2-24d9295f-cd6d9b2e.jpg"
    
    print("=" * 60)
    print("X-Ray Image Comparison Analysis System")
    print("=" * 60)
    
    comparator = XRayImageComparatorCN(image1_path, image2_path)
    
    try:
        print("\n1. Loading images...")
        comparator.load_images()
        
        print("\n2. Aligning images...")
        comparator.align_images()
        
        print("\n3. Computing image differences...")
        comparator.compute_difference()
        
        print("\n4. Detecting lesion regions...")
        comparator.detect_lesions(threshold=30, min_area=100)
        
        print("\n5. Analyzing disease progression...")
        progression_info = comparator.analyze_progression()
        
        print("\n" + "=" * 60)
        print("Analysis Results:")
        print("=" * 60)
        print(f"Progression: {progression_info['progression']}")
        print(f"Severity: {progression_info['severity']}")
        print(f"Lesion Count: {progression_info['lesion_count']}")
        print(f"Total Area: {progression_info['total_area']:.0f} px^2")
        print(f"Avg Intensity: {progression_info['avg_intensity']:.1f}")
        
        print("\n6. Generating comparison visualization...")
        output_dir = r"D:\pyworkspace\evaluation_for_report"
        comparator.visualize_comparison(
            save_path=os.path.join(output_dir, "comparison_result_cn.png")
        )
        
        print("\n7. Generating difference heatmap...")
        comparator.create_difference_heatmap(
            save_path=os.path.join(output_dir, "difference_heatmap_cn.png")
        )
        
        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


#请对main方法中的结果，与data\sceneGraph_json\0a27a7a6-c3bb9cfa-956e0eef-2c19e165-1687ea63_SceneGraph.json文件中描述的内容是否一致？
def compare_diff_result():
    """
    比较main方法的结果与SceneGraph JSON文件中的内容是否一致
    """
    import json
    
    print("\n" + "=" * 60)
    print("比较分析结果与SceneGraph JSON")
    print("=" * 60)
    
    # JSON文件路径
    json_path = r"D:\pyworkspace\evaluation_for_report\data\sceneGraph_json\0a27a7a6-c3bb9cfa-956e0eef-2c19e165-1687ea63_SceneGraph.json"
    
    try:
        # 读取JSON文件
        if not os.path.exists(json_path):
            print(f"错误: JSON文件不存在 - {json_path}")
            print("请先运行download_json.py下载JSON文件")
            return False
        
        with open(json_path, 'r', encoding='utf-8') as f:
            scene_graph = json.load(f)
        
        print(f"成功读取JSON文件: {os.path.basename(json_path)}")
        
        # 分析JSON文件内容
        print("\n1. 分析SceneGraph JSON内容:")
        
        # 提取关键信息
        if 'objects' in scene_graph:
            objects = scene_graph['objects']
            print(f"  - 检测到 {len(objects)} 个对象")
            
            # 统计病灶相关对象
            lesion_objects = []
            for obj in objects:
                if 'type' in obj:
                    obj_type = obj['type'].lower()
                    if any(keyword in obj_type for keyword in ['lesion', 'abnormality', 'nodule', 'mass', 'opacity']):
                        lesion_objects.append(obj)
            
            print(f"  - 病灶相关对象: {len(lesion_objects)}")
            
            for i, obj in enumerate(lesion_objects):
                obj_type = obj.get('type', 'Unknown')
                bbox = obj.get('bbox', 'N/A')
                confidence = obj.get('confidence', 'N/A')
                print(f"    {i+1}. {obj_type} - 置信度: {confidence} - 边界框: {bbox}")
        else:
            print("  - JSON文件中未找到'objects'字段")
        
        # 分析main方法的结果（这里需要重新运行分析以获取最新结果）
        print("\n2. 分析main方法的结果:")
        
        # 重新运行分析获取结果
        image1_path = r"D:\pyworkspace\evaluation_for_report\data\jpg\p11000566\s50230446\2e4678a5-e646a648-d6265814-63b082e3-d14f047a.jpg"
        image2_path = r"D:\pyworkspace\evaluation_for_report\data\jpg\p11000566\s50252971\99eb5ea2-76aff341-b0db7fe2-24d9295f-cd6d9b2e.jpg"
        
        comparator = XRayImageComparatorCN(image1_path, image2_path)
        comparator.load_images()
        comparator.align_images()
        comparator.compute_difference()
        lesion_boxes = comparator.detect_lesions(threshold=30, min_area=100)
        progression_info = comparator.analyze_progression()
        
        print(f"  - 检测到 {len(lesion_boxes)} 个病灶区域")
        print(f"  - 病情进展: {progression_info['progression']}")
        print(f"  - 严重程度: {progression_info['severity']}")
        print(f"  - 病灶总面积: {progression_info['total_area']:.0f} 像素平方")
        print(f"  - 平均强度: {progression_info['avg_intensity']:.1f}")
        
        # 比较分析
        print("\n3. 比较结果:")
        
        if 'objects' in scene_graph:
            json_lesion_count = len([obj for obj in scene_graph['objects'] if 
                                   'type' in obj and any(keyword in obj['type'].lower() for keyword in 
                                   ['lesion', 'abnormality', 'nodule', 'mass', 'opacity'])])
            
            detected_lesion_count = len(lesion_boxes)
            
            print(f"  - JSON中的病灶数量: {json_lesion_count}")
            print(f"  - 检测到的病灶数量: {detected_lesion_count}")
            
            if json_lesion_count == detected_lesion_count:
                print("  [√] 病灶数量一致")
            else:
                print("  [×] 病灶数量不一致")
            
            # 比较边界框位置（如果JSON中有边界框信息）
            if lesion_boxes and 'objects' in scene_graph:
                print("\n  边界框位置比较:")
                for i, (detected_box, json_obj) in enumerate(zip(lesion_boxes[:json_lesion_count], 
                                                              [obj for obj in scene_graph['objects'] if 
                                                               'type' in obj and any(keyword in obj['type'].lower() for keyword in 
                                                               ['lesion', 'abnormality', 'nodule', 'mass', 'opacity'])])):
                    det_x, det_y, det_w, det_h = detected_box['box']
                    json_bbox = json_obj.get('bbox', [0, 0, 0, 0])
                    
                    # 计算中心坐标差异
                    det_center = (det_x + det_w/2, det_y + det_h/2)
                    json_center = (json_bbox[0] + json_bbox[2]/2, json_bbox[1] + json_bbox[3]/2)
                    
                    distance = np.sqrt((det_center[0] - json_center[0])**2 + (det_center[1] - json_center[1])**2)
                    
                    print(f"    病灶 {i+1}: 中心距离 = {distance:.1f} 像素")
                    if distance < 50:  # 阈值可调整
                        print(f"    [√] 位置基本一致")
                    else:
                        print(f"    [×] 位置差异较大")
        
        print("\n4. 结论:")
        print("  - 分析完成，结果已比较")
        print("  - 详细比较结果见上面输出")
        
        return True
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
    compare_diff_result()
