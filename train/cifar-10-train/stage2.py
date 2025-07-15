import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# 超参
LEARNING_RATE = 1e-3  # 微调 但是感觉他也不能太微
BATCH_SIZE = 128
NUM_EPOCHS = 30
# WEIGHT_DECAY = 1e-4 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER_WEIGHTS_PATH = "./model/cifar10_rotnet_encoder.pth"

print(f"使用设备: {DEVICE}")


# 训练集使用强数据增强来提升模型泛化能力
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),       # 随机裁剪
    transforms.RandomHorizontalFlip(),          # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 验证集不需要数据增强，以保证评估结果的一致性
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# 和原文完全一样的分类头
class MLPHeadFromPaper(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, output_dim=10):
        super(MLPHeadFromPaper, self).__init__()
        self.head = nn.Sequential(
            # 第1个隐藏层
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # 对全连接层的输出使用BatchNorm1d
            nn.ReLU(inplace=True),
            
            # 第2个隐藏层
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # 输出层
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.head(x)


class CIFAR10Classifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super(CIFAR10Classifier, self).__init__()
        self.encoder = encoder
        num_features = 512 # ResNet-18的输出特征数
        
        # 使用更强大的分类头
        self.classifier = MLPHeadFromPaper(input_dim=num_features, output_dim=num_classes)
        
    def forward(self, x):
        features = self.encoder(x)
        outputs = self.classifier(features)
        return outputs

# 加载预训练的Encoder
pretrained_encoder = models.resnet18(weights=None)
pretrained_encoder.fc = nn.Identity()

choice = input('是否加载权重?[Y/n]')
if choice.lower() == 'y':
    print(f"正在从 {ENCODER_WEIGHTS_PATH} 加载权重...")
    pretrained_encoder.load_state_dict(torch.load(ENCODER_WEIGHTS_PATH, map_location=DEVICE))
    print("权重加载成功!")
else:
    print('不预加载权重')

# 不再冻结Encoder，而是采用全网络微调
# for param in pretrained_encoder.parameters():
#     param.requires_grad = False

model = CIFAR10Classifier(encoder=pretrained_encoder, num_classes=10).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam([
    {'params': model.encoder.parameters(), 'lr': LEARNING_RATE / 10},   # Encoder使用较小的学习率
    {'params': model.classifier.parameters(), 'lr': LEARNING_RATE}      # 分类头使用较大的学习率
])

# 学习率调度器
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# 训练循环
for epoch in range(NUM_EPOCHS):
    model.train() # 将整个模型设为训练模式
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    
    progress_bar = tqdm(train_loader, desc="训练中")
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
        
        progress_bar.set_postfix(loss=running_loss/total_train, acc=f"{(100*correct_train/total_train):.2f}%")

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
    
    # 验证循环
    model.eval() 
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="验证中"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    val_loss_avg = val_loss / len(val_loader)
    val_acc = 100 * correct_val / total_val
    print(f"验证损失: {val_loss_avg:.4f} | 验证准确率: {val_acc:.2f}%")
    
    # # --- 【改动 6】在每个epoch结束后更新学习率 ---
    # scheduler.step()
    # print(f"当前学习率: {scheduler.get_last_lr()}")

print("\n下游任务训练完成!")
