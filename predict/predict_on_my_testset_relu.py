import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("--- Model Testing Script (Enhanced Analysis Version) ---")
print("For each test image, this script will test all 4 rotations.")


MODEL_PATH = './model/mydata_rotnet_relu.pth' 
TEST_IMAGE_DIR = './data/test_images/'
OUTPUT_DIR = './results_enhanced/' 
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Loading model weights from: {MODEL_PATH}")
print(f"Testing images from directory: {TEST_IMAGE_DIR}")
print(f"Enhanced results will be saved to: {OUTPUT_DIR}")

class RotationPredictionModel(nn.Module):
    def __init__(self, num_classes=4):
        super(RotationPredictionModel, self).__init__()
        self.encoder = models.resnet18(weights=None) 
        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.classifier(features)
        return outputs

# --- 3. Load Model ---
model = RotationPredictionModel(num_classes=4).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'. Please check the path.")
    exit()

model.eval()
print("Model loaded successfully and set to evaluation mode.")

# --- 4. Define Image Transformation for Testing ---
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 5. Iterate Through Test Folder, Augment, Predict, Correct, and Save ---
if not os.path.isdir(TEST_IMAGE_DIR):
    print(f"Error: Test image directory not found at '{TEST_IMAGE_DIR}'.")
    exit()
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_images = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
if not test_images:
    print(f"Warning: No image files found in '{TEST_IMAGE_DIR}'.")
    exit()
    
angles = [0, 90, 180, 270]

# 初始化总计数器
total_tests = 0
correct_predictions = 0
# 新增: 初始化分角度统计的计数器, 格式为 {角度: [正确数, 总数]}
angle_stats = {angle: [0, 0] for angle in angles}

# 外层循环：遍历每个原始图片文件
for image_name in test_images:
    image_path = os.path.join(TEST_IMAGE_DIR, image_name)
    try:
        # 先用基础方式打开图片
        img = Image.open(image_path).convert("RGB")
        
        # 关键修复: 使用 ImageOps.exif_transpose 来根据EXIF信息自动旋转图片
        # 这可以确保代码加载的图片方向和你在图片查看器中看到的方向一致
        original_image = ImageOps.exif_transpose(img)
        
        # 内层循环：对每个原始图片进行四次旋转测试
        for test_rotation_angle in angles:
            print(f"\n--- Testing File: {image_name}, Rotated by: {test_rotation_angle} deg ---")
            
            # 1. 创建本次测试的输入图片
            input_image = original_image.rotate(test_rotation_angle, expand=True)
            
            # 2. 预处理并进行预测
            input_tensor = test_transform(input_image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(input_tensor)
            
            _, predicted_idx = torch.max(output.data, 1)
            predicted_angle = angles[predicted_idx.item()]
            
            print(f"  -> Model Prediction: {predicted_angle} degrees")

            # 3. 判断预测是否正确并更新计数器
            is_correct = (predicted_angle == test_rotation_angle)
            correctness_str = "Correct" if is_correct else "INCORRECT"
            print(f"  -> Result: {correctness_str}")
            
            # 更新总计数器
            total_tests += 1
            # 更新分角度计数器
            angle_stats[test_rotation_angle][1] += 1 # 该角度的总数+1
            if is_correct:
                correct_predictions += 1
                angle_stats[test_rotation_angle][0] += 1 # 该角度的正确数+1

            # 4. 校正图片 (基于模型的预测)
            correction_for_input = -predicted_angle
            corrected_image = input_image.rotate(correction_for_input, expand=True)
            
            # 5. 生成并保存对比图
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(input_image)
            axes[0].set_title(f"Input (Manually Rotated by {test_rotation_angle} deg)")
            axes[0].axis('off')
            
            axes[1].imshow(corrected_image)
            axes[1].set_title(f"Model Corrected (Predicted {predicted_angle} deg)\nResult: {correctness_str}")
            axes[1].axis('off')
            
            fig.suptitle(f"File: {image_name}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            base_name, ext = os.path.splitext(image_name)
            output_filename = f"comparison_{base_name}_test{test_rotation_angle}{ext}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            plt.savefig(output_path)
            plt.close(fig) 
            print(f"  -> Comparison chart saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred while processing {image_name}: {e}")

print("\n--- All test images and their rotations processed. ---")

# --- 6. 打印最终的测试总结 ---
if total_tests > 0:
    final_accuracy = (correct_predictions / total_tests) * 100
    print("\n--- Final Test Summary ---")
    print(f"Total tests performed: {total_tests}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Overall Accuracy: {final_accuracy:.2f}%")
    print("\n--- Detailed Per-Angle Accuracy ---")
    for angle, (correct, total) in angle_stats.items():
        if total > 0:
            accuracy = (correct / total) * 100
            print(f"Angle {angle:<3} deg: {correct:>3}/{total:<3} correct -> Accuracy: {accuracy:.2f}%")
        else:
            print(f"Angle {angle:<3} deg: No tests performed.")
else:
    print("\n--- No tests were run. ---")

