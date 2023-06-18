import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# 设置数据路径
#data_path = './Brain/3-BrainTumor/'
data_path = './Brain/3-BrainTumor/enhance/'
#image_dir = 'iexp/'
#mask_dir = 'mexp/'
image_dir = 'augmented_images/'
mask_dir = 'augmented_masks/'

# 读取所有图像和标签
X = []
y = []
for filename in os.listdir(os.path.join(data_path, image_dir)):
    img = cv2.imread(os.path.join(data_path, image_dir, filename), cv2.IMREAD_GRAYSCALE)
    X.append(img)

    mask = cv2.imread(os.path.join(data_path, mask_dir, filename), cv2.IMREAD_GRAYSCALE)
    y.append(mask)

# 打印文件名和维度
#    print("文件名: ", filename, img.shape)


# 分割数据集
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class DoubleConv(nn.Module):  # 连续两次卷积、归一化、激活的操作，整合为一个网络层
   #(convolution => [BN] => ReLU) * 2

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):  #采用最大池化实现降采样，并且和DoubleConv函数组合构成一个网络层
   

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):  #使用上采样实现升采样，并和DoubleConv函数组合构成一个网络层。
    # 在升采样时，若bilinear参数为True，则使用双线性插值进行上采样；否则，使用转置卷积进行上采样。
    # 在将两个张量连接时，需要对它们的尺寸进行padding操作以使得尺度匹配。
   

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if you want to use bilinear interpolation, uncomment the line below
        # and comment the two lines after t_conv = ...
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):  #卷积操作，将U-Net的输出结果转换为指定的通道数
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):  #x1到x5是不同层次降采样得到的特征图，分别作为上采样时的辅助输入。
    # 在最后输出结果时，使用OutConv进行通道数转换，返回最终的分类概率张量。
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# 构造数据集
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).unsqueeze(0).float(), torch.from_numpy(self.y[idx]).unsqueeze(0).float()

# 定义训练函数
def train(model, train_loader, valid_loader, num_epochs, criterion, optimizer):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print('Epoch: {}, Training Loss: {}'.format(epoch+1, train_loss))

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for data, target in valid_loader:
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        print('Epoch: {}, Validation Loss: {}'.format(epoch+1, valid_loss))
        torch.save(model.state_dict(), 'model.pth')

# 准备数据
input_shape = (1, 512, 512)

# 创建训练集、验证集和测试集
train_dataset = MyDataset(X_train, y_train)
valid_dataset = MyDataset(X_val, y_val)
test_dataset = MyDataset(X_test, y_test)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 初始化模型
model = UNet(input_shape[0], 1, True)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
train(model, train_loader, valid_loader, num_epochs=10, criterion=criterion, optimizer=optimizer)

# 测试模型
model.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()

        # 修改部分：
        pred = (torch.sigmoid(output) > 0.5).float()  # 对sigmoid后的模型输出进行二值化
        # 统计 mask 中白色部分的像素数量
        total += torch.sum(target > 0).item()
        # 比较预测结果和真实标签，并对mask内的像素点进行统计
        correct += ((pred == target) * target).sum().item()

test_loss /= len(test_loader)
accuracy = correct / total
print("Test Loss:", test_loss)
print("Accuracy:", accuracy)

