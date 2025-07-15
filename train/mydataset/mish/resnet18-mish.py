#!/usr/bin/env python3
"""
RotNet (ResNet-18 Mish版) 训练脚本 - 使用余弦退火学习率调度器。
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, models
from PIL import Image, ImageOps
import random
import os
from tqdm import tqdm

LEARNING_RATE = 3e-4
BATCH_SIZE = 32
NUM_EPOCHS = 30 
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224
DROP_RATE = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIR = './data/image/'
MODEL_SAVE_DIR = './model/'



def replace_relu_with_mish(model):
    """
    递归地将所有 nn.ReLU 层替换为 nn.Mish 层。
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_relu_with_mish(module)
        if isinstance(module, nn.ReLU):
            setattr(model, name, nn.Mish())

# 数据集
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        try:
            self.img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
        except FileNotFoundError:
            print(f"错误: 找不到目录 '{img_dir}'。")
            self.img_paths = []
        if not self.img_paths:
            print(f"警告: 在目录 '{img_dir}' 中没有找到任何图片文件。")
        else:
            print(f"在目录 '{img_dir}' 中找到 {len(self.img_paths)} 张图片。")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = ImageOps.exif_transpose(image)
            if self.transform:
                image = self.transform(image)
            return image, 0
        except Exception as e:
            print(f"错误: 无法加载或处理图片 {img_path}。错误信息: {e}")
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), 0

class RandomRotation:
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, img):
        angle_choice = random.choice(self.angles)
        rotated_img = img.rotate(angle_choice, expand=True)
        label = self.angles.index(angle_choice)
        return rotated_img, label

class RotationDataset(Dataset):
    def __init__(self, underlying_dataset, final_transform=None):
        self.underlying_dataset = underlying_dataset
        self.rotation_transform = RandomRotation()
        self.final_transform = final_transform

    def __len__(self):
        return len(self.underlying_dataset)

    def __getitem__(self, idx):
        original_img, _ = self.underlying_dataset[idx]
        rotated_img, rotation_label = self.rotation_transform(original_img)
        if self.final_transform:
            final_img = self.final_transform(rotated_img)
        return final_img, torch.tensor(rotation_label, dtype=torch.long)


# 数据增强 自己的数据集太薄弱了，不增强效果不好
train_base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
])

val_base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
])

final_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载和准备数据 
underlying_train_dataset = CustomImageDataset(img_dir=IMAGE_DIR, transform=train_base_transform)
underlying_val_dataset = CustomImageDataset(img_dir=IMAGE_DIR, transform=val_base_transform)

if len(underlying_train_dataset) == 0:
    raise ValueError(f"错误: 在 '{IMAGE_DIR}' 中没有找到任何图片。")

train_size = int(0.8 * len(underlying_train_dataset))
val_size = len(underlying_train_dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_indices, val_indices = random_split(range(len(underlying_train_dataset)), [train_size, val_size], generator=generator)

train_subset = Subset(underlying_train_dataset, train_indices)
val_subset = Subset(underlying_val_dataset, val_indices)
print(f"数据集已分割: {len(train_subset)} 张用于训练, {len(val_subset)} 张用于验证。")

train_dataset = RotationDataset(train_subset, final_transform=final_transform)
val_dataset = RotationDataset(val_subset, final_transform=final_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# 模型定义
class RotationPredictionModel(nn.Module):
    def __init__(self, num_classes=4):
        super(RotationPredictionModel, self).__init__()
        self.encoder = models.resnet18(weights='IMAGENET1K_V1')
        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        
        # 根据是否测试过拟合来调整Dropout率
        dropout_rate = DROP_RATE
        print(f"分类头Dropout率设置为: {dropout_rate}")
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.classifier(features)
        return outputs

# --- 6. 初始化模型、损失函数和优化器 ---
model = RotationPredictionModel(num_classes=4).to(DEVICE)
replace_relu_with_mish(model)

criterion = nn.CrossEntropyLoss()
# 根据是否测试过拟合来调整权重衰减
wd = WEIGHT_DECAY
print(f"优化器权重衰减设置为: {wd}")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=wd)

# 使用余弦退火学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# 训练和验证
best_val_acc = 0.0
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    progress_bar = tqdm(train_loader, desc="训练中", colour="green")
    for images, labels in progress_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        progress_bar.set_postfix(loss=loss.item())

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")

    model.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="验证中", colour="cyan")
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct_val / total_val
    print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")

    # 动态调整更新学习率
    # CosineAnnealingLR在每个epoch后调用step()，无需参数
    scheduler.step()
    print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(MODEL_SAVE_DIR, "mydata_rotnet_mish.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"🎉 新的最佳验证准确率: {best_val_acc:.2f}%。模型已保存至: {best_model_path}")


print("\n训练完成!")
print(f"整个训练过程中最佳的验证准确率为: {best_val_acc:.2f}%")

# 保存最后一轮训练的encoder用于二阶段的分类器训练
print("\n正在提取并保存最终的 Encoder 权重以用于下游任务")
ENCODER_OUTPUT_PATH = "./model/mydata_rotnet_mish_encoder.pth"
torch.save(model.encoder.state_dict(), ENCODER_OUTPUT_PATH)
print(f"Encoder 权重已保存至: '{ENCODER_OUTPUT_PATH}'")
