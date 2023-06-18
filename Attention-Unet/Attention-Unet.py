import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 定义卷积块


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# 注意力板块


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.conv(x)
        return x * att

# 上采样卷积块


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.upconv(x)

# AttentionUNet类


class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUNet, self).__init__()
        self.down1 = ConvBlock(in_channels, 64)
        self.bn1 = nn.BatchNorm2d(64)  # 添加BatchNorm2d层
        self.down2 = ConvBlock(64, 128)
        self.bn2 = nn.BatchNorm2d(128)  # 添加BatchNorm2d层
        self.down3 = ConvBlock(128, 256)
        self.bn3 = nn.BatchNorm2d(256)
        self.down4 = ConvBlock(256, 512)
        self.bn4 = nn.BatchNorm2d(512) 
        self.bottom = ConvBlock(512, 1024)
        self.up4 = UpConvBlock(1024, 512)
        self.att4 = AttentionBlock(1024)
        self.up3 = UpConvBlock(1024, 256)
        self.att3 = AttentionBlock(512)
        self.up2 = UpConvBlock(512, 128)
        self.att2 = AttentionBlock(256)
        self.up1 = UpConvBlock(256, 64)
        self.att1 = AttentionBlock(128)
        self.outconv = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        down1 = self.down1(x)
        down1 = self.bn1(down1)  
        down2 = self.down2(nn.functional.max_pool2d(down1, 2))
        down2 = self.bn2(down2)  
        down3 = self.down3(nn.functional.max_pool2d(down2, 2))
        down3 = self.bn3(down3)
        down4 = self.down4(nn.functional.max_pool2d(down3, 2))
        down4 = self.bn4(down4)
        bottom = self.bottom(nn.functional.max_pool2d(down4, 2))
        up4 = self.up4(bottom)
        att4 = self.att4(torch.cat((down4, up4), dim=1))
        up3 = self.up3(att4)
        att3 = self.att3(torch.cat((down3, up3), dim=1))
        up2 = self.up2(att3)
        att2 = self.att2(torch.cat((down2, up2), dim=1))
        up1 = self.up1(att2)
        att1 = self.att1(torch.cat((down1, up1), dim=1))
        out = self.outconv(att1)
        return out


class BrainMRI_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.image_filenames = os.listdir(self.image_dir)
        self.mask_filenames = os.listdir(self.mask_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(image_path).convert('L')  # 转为灰度图像
        mask = Image.open(mask_path).convert('L')  # 转为灰度图像

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def calculate_pixel_accuracy(outputs, masks):
    predicted_labels = (outputs > 0.5).float()
    # print('predicted_labels',predicted_labels)
    # print('masks',masks)
    masks = masks == torch.max(masks)
    correct = (predicted_labels == masks).float().sum()
    # print('correct',correct)
    total = masks.numel()
    accuracy = (correct / total).item()
    return accuracy


def calculate_dice_score(outputs, masks):
    smooth = 1e-6
    predicted_labels = (outputs > 0.5).float()
    intersection = (predicted_labels * masks).sum()
    union = predicted_labels.sum() + masks.sum()
    dice_score = ((2 * intersection + smooth) / (union + smooth)).item()
    return dice_score


def calculate_metrics(outputs, masks):
    # print(outputs)
    # print(masks)
    predictions = outputs > 0.5
    masks = masks == torch.max(masks)
    # print('predictions:',predictions)
    # print('masks:',masks)
    # TP = ((predictions==1)+(masks==1))==2
    # print('TP:',TP)
    # FN = ((predictions==0)+(masks==1))==2
    # TN = ((predictions==0)+(masks==0))==2
    # FP = ((predictions==1)+(masks==0))==2
    # SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)    
    # SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    TP = ((predictions == 1) & (masks == 1)).sum().item()
    TN = ((predictions == 0) & (masks == 0)).sum().item()
    FP = ((predictions == 1) & (masks == 0)).sum().item()
    FN = ((predictions == 0) & (masks == 1)).sum().item()
    total = masks.numel()
    return TP/total, TN/total, FP/total, FN/total
    # return SE, SP


def dice_loss(outputs, masks):
    smooth = 1e-6
    predicted_labels = (outputs > 0.5).float()
    intersection = (predicted_labels * masks).sum()
    union = predicted_labels.sum() + masks.sum()
    dice_score = (2 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_score
    return dice_loss


data_root = 'D:\\testcode_py\\enhanced'  # 数据集根目录

# 设置随机种子以确保可重复性
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 数据预处理和增强
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 将图像调整为128x128像素
    transforms.ToTensor(),
    # 如果需要进行归一化等其他预处理操作，可以在这里添加
    # transforms.Normalize((0.5), (0.5))
])

# 创建数据集实例
dataset = BrainMRI_Dataset(data_root, transform=transform)

# 划分训练集和验证集，可以根据需要调整划分比例
val_split = 0.2
val_size = int(val_split * len(dataset))
train_size = len(dataset) - val_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 创建数据加载器
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 查看dataloader中每一个批次的数据维度
dataiter = iter(train_loader)
batch = next(dataiter)
inputs, labels = batch
print("inputs.shape:", inputs.shape)
print("labels.shape:", labels.shape)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = AttentionUNet(in_channels=1, out_channels=1)  # 输入和输出通道数都为1
model.to(device)

criterion = nn.BCELoss()  # 使用二元交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

num_epochs = 5
save_path = './model_params.pth'  # 模型参数保存的文件路径
if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path))
    print('successful load weight!')
else:
    print('not successful load weight')
# 训练循环
for epoch in range(num_epochs):
    batch_index = 0
    model.train()
    train_loss = 0.0
    # 初始化像素精确度和Dice分数变量
    train_pixel_accuracy = 0.0
    train_dice_score = 0.0
    
    for images, masks in train_loader:
        
        images = images.to(device)
        masks = masks.to(device)
        # print('train images device:', images.device)
        optimizer.zero_grad()
        outputs = torch.sigmoid(model(images))
        masks = torch.sigmoid(masks)
        # print(outputs)
        # print(masks==0)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        # 计算像素精确度和Dice分数
        train_pixel_accuracy += calculate_pixel_accuracy(outputs, masks) * images.size(0)
        train_dice_score += calculate_dice_score(outputs, masks) * images.size(0)

        train_loss += loss.item() * images.size(0)
        batch_index = batch_index + 1
        if(batch_index % 100 == 0):
            print(train_loss)
            print(train_pixel_accuracy)
            print(train_dice_score)

    torch.save(model.state_dict(), save_path)

    train_loss /= len(train_dataset)
    train_pixel_accuracy /= len(train_dataset)
    train_dice_score /= len(train_dataset)
    
    # 在验证集上评估模型性能
    model.eval()
    val_loss = 0.0
    val_pixel_accuracy = 0.0
    val_dice_score = 0.0
    TP_total, TN_total, FP_total, FN_total = 0, 0, 0, 0

    with torch.no_grad():
        for images, masks in val_loader:
            
            images = images.to(device)
            masks = masks.to(device)
            # print('val images device:', images.device)
            outputs = torch.sigmoid(model(images))
            masks = torch.sigmoid(masks)
            # print(torch.mean(outputs))
            # print(torch.mean(masks))
            loss = criterion(outputs, masks)

            val_loss += loss.item() * images.size(0)
            val_pixel_accuracy += calculate_pixel_accuracy(outputs, masks) * images.size(0)
            val_dice_score += calculate_dice_score(outputs, masks) * images.size(0)
            # TP, TN, FP, FN = calculate_metrics(outputs, masks)
            # print(val_dice_score)
            # print(outputs==0)
            TP, TN, FP, FN = calculate_metrics(outputs, masks) 
            TP_total += TP
            TN_total += TN
            FP_total += FP
            FN_total += FN

    val_loss /= len(val_dataset)
    val_pixel_accuracy /= len(val_dataset)
    val_dice_score /= len(val_dataset)
    precision = TP_total / (TP_total + FP_total)
    recall = TP_total / (TP_total + FN_total)
    specificity = TN_total / (TN_total + FP_total)
    F1_Score = 2 * precision * recall / (precision + recall)
    TP_total /= len(val_dataset)
    TN_total /= len(val_dataset)
    FP_total /= len(val_dataset)
    FN_total /= len(val_dataset)
    # accuracy = (TP_total + TN_total) / (TP_total + TN_total + FP_total + FN_total)

    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    print(f'Epoch {epoch+1}/{num_epochs}, train_pixel_accuracy: {train_pixel_accuracy:.4f},  val_pixel_accuracy: {val_pixel_accuracy:.4f}')
    print(f'Epoch {epoch+1}/{num_epochs}, train_dice_score: {train_dice_score:.4f}, val_dice_score: {val_dice_score:4f}')
    print(f'Precision: {precision:.6f}, Recall: {recall:.6f}, Specificity:{specificity:.6f}, F1-score:{F1_Score:.6f}')
    print(f'TP: {TP_total:.4f},TN: {TN_total:.4f}, FP: {FP_total:.4f}, FN: {FN_total:.20f}')