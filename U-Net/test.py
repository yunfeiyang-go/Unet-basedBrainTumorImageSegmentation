import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# 设置数据路径
image_folder = './Brain/3-BrainTumor/enhance/augmented_images/'
mask_folder = './Brain/3-BrainTumor/enhance/augmented_masks/'
#data_path = './bioalgorithm/Brain/3-BrainTumor/enhance/'
#image_dir = 'augmented_images/'
#mask_dir = 'augmented_masks/'

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

# 准备数据
input_shape = (1, 512, 512)

import torchvision.transforms as transforms
from PIL import Image

# 设置GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = UNet(input_shape[0], 1, True).to(device)
model.load_state_dict(torch.load("./Brain/U-Net/model.pth"))

# 定义transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# 定义函数来计算 TP, TN, FP, FN 和 Dice 值
def calculate_metrics(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    true_positive = ((pred >= 0.5) & (target == 1)).sum().item()
    true_negative = ((pred < 0.5) & (target == 0)).sum().item()
    false_positive = ((pred >= 0.5) & (target == 0)).sum().item()   # 只考虑真实标签中的阳性像素点
    false_negative = ((pred < 0.5) & (target == 1)).sum().item()
    return true_positive, true_negative, false_positive, false_negative

# 定义函数来对单张图片进行预测和计算指标
def predict_image(image_path, mask_path):
    # 打开图片和mask文件
    image = Image.open(image_path).convert('L')
    mask = Image.open(mask_path).convert('L')

    # 对图片进行transform
    image = transform(image).unsqueeze(0).to(device)
    mask = transform(mask).unsqueeze(0).to(device)

    # 进行预测
    with torch.no_grad():
        output = model(image)

    # 计算指标
    true_positive, true_negative, false_positive, false_negative = calculate_metrics(output, mask)

    # 计算mask区域像素点总个数
    total_pixels = (mask == 1).sum().item()

    # 返回结果
    return total_pixels, true_negative, true_positive, false_negative, false_positive

# 定义函数来遍历文件夹并调用 predict_image 函数
def predict_folder(image_folder, mask_folder):
    image_files = os.listdir(image_folder)
    mask_files = os.listdir(mask_folder)
    
    dice_scores = []      # 存储所有图像的 Dice 分数
    accuracies = []       # 存储所有图像的预测准确率
    precisions = []       # 存储所有图像的 Precision 值
    recalls = []          # 存储所有图像的 Recall 值
    specificities = []    # 存储所有图像的 Specificity 值
    f1_scores = []        # 存储所有图像的 F1-score 值

    for i in range(len(image_files)):
        image_path = os.path.join(image_folder, image_files[i])
        mask_path = os.path.join(mask_folder, mask_files[i])
        total_pixels, true_negative, true_positive, false_negative, false_positive = predict_image(image_path, mask_path)
        
        # 计算dice、预测准确率、Precision、Recall、Specificity 和 F1-score，并将其添加到列表中
        dice_score = (2 * true_positive) / (2 * true_positive + false_positive + false_negative) 
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        precision = true_positive / (true_positive + false_positive + 1e-6)
        recall = true_positive / (true_positive + false_negative + 1e-6)
        specificity = true_negative / (true_negative + false_positive + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        
        dice_scores.append(dice_score)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        f1_scores.append(f1_score)
        
        # 输出结果
        print("第", i+1, "张图片的结果：")
        print("mask像素点共有：",total_pixels)
        print("TP,TN,FP,FN:",true_positive, ", " , true_negative, ", " , false_positive, ", " , false_negative)
        print("Dice值: ", dice_score)
        print("预测准确率: {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("Specificity: {:.4f}".format(specificity))
        print("F1-score: {:.4f}".format(f1_score))
        print("\n")
    
    mean_dice = round(sum(dice_scores) / len(dice_scores), 4)
    mean_accuracy = round(sum(accuracies) / len(accuracies), 4)
    mean_precision = round(sum(precisions) / len(precisions), 4)
    mean_recall = round(sum(recalls) / len(recalls), 4)
    mean_specificity = round(sum(specificities) / len(specificities), 4)
    mean_f1_score = round(sum(f1_scores) / len(f1_scores), 4)
    
    # 输出平均结果
    print("所有图像的平均 Dice 值为：", mean_dice)
    print("所有图像的平均预测准确率为：{:.4f}".format(mean_accuracy))
    print("所有图像的平均 Precision 值为：{:.4f}".format(mean_precision))
    print("所有图像的平均 Recall 值为：{:.4f}".format(mean_recall))
    print("所有图像的平均 Specificity 值为：{:.4f}".format(mean_specificity))
    print("所有图像的平均 F1-score 值为：{:.4f}".format(mean_f1_score))

import sys

# 在屏幕和文件中输出
sys.stdout = open('./Brain/U-Net/test.txt', 'w')

# 调用 predict_folder 函数对文件夹内的图片进行预测并输出结果
predict_folder(image_folder, mask_folder)

# 恢复标准输出
sys.stdout = sys.__stdout__


