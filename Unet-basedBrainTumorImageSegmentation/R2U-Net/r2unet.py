import torch
import torch.nn as nn
from torchvision import transforms  # 是一个常用的图片变换类
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import torchvision.transforms.functional as TF
from torch.utils.data import SubsetRandomSampler
from PIL import Image
from time import time
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt

class TumorDataset(Dataset):
    def __init__(self, images_dir,masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.name=os.listdir(self.masks_dir)
        self.default_transformation = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        total_files = len(os.listdir(self.images_dir))
        return total_files

    def __getitem__(self, index):
        mask_name = self.name[index]

        image_path = os.path.join(self.images_dir,mask_name)
        mask_path = os.path.join(self.masks_dir,mask_name)

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = self.default_transformation(image)
        mask = self.default_transformation(mask)

        return image,mask

# define model
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, padding_mode='reflect',bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1

class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class R2UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, t=2):
        super(R2UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)


        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)

        return d1

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = (SR > threshold).float()
    Inter = (SR * GT).sum()
    DC = ((2*Inter)/(SR.sum()+GT.sum())).item()

    return DC

def get_accuracy(SR,GT,threshold=0.5):
    SR = (SR > threshold).float()
    GT=GT==torch.max(GT)
    corr = (SR==GT).float().sum()
    total = GT.numel()
    acc = (corr/total).item()

    return acc

def metrix(mask_pred,mask,threshold=0.5):
    predictions = mask_pred>0.5
    mask=mask==torch.max(mask)
    TP = ((predictions == 1 ) & (mask == 1)).sum().item()
    TN = ((predictions == 0 ) & (mask == 0)).sum().item()
    FP = ((predictions == 1 ) & (mask == 0)).sum().item()
    FN = ((predictions == 0 ) & (mask == 1)).sum().item()
    total = mask.numel()
    return TP/total,TN/total,FP/total,FN/total

def loss_graph(loss_list, save_plot=None):

    plt.figure(figsize=(20, 10))
    plt.title('Loss Function Over Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    line = plt.plot(loss_list, marker='o')
    plt.legend((line), ('Loss Value',), loc=1)
    if save_plot:
        plt.savefig(save_plot)
    plt.show()

#loading data
BATCH_SIZE = 8
EPOCHS = 20
TEST_SPLIT = 0.2

#dir = 'C:\\Users\\WANG\\Desktop\\3-BrainTumor'
dir = '/home/ug2020/ug520111910221/bigdata/brain/3-BrainTumor/enhance'
#image_dir= 'C:\\Users\\WANG\\Desktop\\3-BrainTumor-2\\images'
image_dir = '/home/ug2020/ug520111910221/bigdata/brain/3-BrainTumor/enhance/images'
#mask_dir='C:\\Users\\WANG\\Desktop\\3-BrainTumor-2\\masks'
mask_dir = '/home/ug2020/ug520111910221/bigdata/brain/3-BrainTumor/enhance/masks'
save_path='/home/ug2020/ug520111910221/bigdata/brain/train_image-new'
save_path_2='/home/ug2020/ug520111910221/bigdata/brain/test_image-neiw'
data=dict()
dataset = TumorDataset(image_dir,mask_dir)
indices = list(range(len(dataset)))
split = int(np.floor(TEST_SPLIT * len(dataset)))
train_indices , test_indices = indices[split:], indices[:split]
train_sampler, test_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices)

trainloader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, sampler=train_sampler)
testloader = torch.utils.data.DataLoader(dataset, 1, sampler=test_sampler)

testlen=len(dataset)*TEST_SPLIT
trainlen = len(dataset)-testlen
# model
model= R2UNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 将数据放在GPU上跑所需要的代码
model.to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # s随机梯度下降优化
model_path = '/home/ug2020/ug520111910221/bigdata/brain/BraTS_new.pth'
#model_path= 'C:\\Users\\WANG\\CLionProjects\\assignment7\\imageseg\\BraTS.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print('successful load weight!')
else:
    print('not successful load weight')

for epoch in range(EPOCHS):
    epoch_loss=0
    for i,(image,mask) in enumerate(trainloader):
        image,mask=image.to(device),mask.to(device)
        mask_pred=model(image)
        train_loss=loss_fn(mask_pred,mask)
        epoch_loss+=train_loss.item()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if i%5==0:
            print('train_loss:',train_loss.item())

        if i%50==0:
            torch.save(model.state_dict(),model_path)

        _image=image[0]
        _mask=mask[0]
        _mask_pred=mask_pred[0]

        img=torch.stack([_image,_mask,_mask_pred],dim=0)
        save_image(img,f'{save_path}/{i}.png')

    epoch_loss=epoch_loss/trainlen
    print('epoch:',epoch,',epoch_loss:',epoch_loss)





def test(testloader,len,threshold=0.5):
    model.eval()
    DC = 0.00
    test_loss=0.00
    acc = 0.00	# Accuracy
    TP = 0.00
    TN = 0.00	
    FP = 0.00	
    FN = 0.00
    with torch.no_grad():
        for i, (image,mask) in enumerate(testloader):
            image = image.view((1, 1, 512, 512)).to(device)
            mask = mask.to(device)
            mask_pred = model(image).to(device)
            test_loss += loss_fn(mask_pred, mask).item()
	    mask_pred_1=torch.sigmoid(mask_pred)
            mask_1=torch.sigmoid(mask)
            DC=get_DC(mask_pred_1,mask_1,threshold)
            acc += get_accuracy(mask_pred_1,mask_1)
            a,b,c,d = metrix(mask_pred_1,mask_1)
            TP += a
            TN += b
            FP += c
            FN += d

            if i%5==0:
                print('test_loss:',test_loss)
                print('DC:',DC)
                
            _image=image[0]
            _mask=mask[0]
            _mask_pred=mask_pred[0]
    
            img=torch.stack([_image,_mask,_mask_pred],dim=0)
            save_image(img,f'{save_path_2}/{i}.png')

    # Calculating the mean score for the whole test dataset.
    DC = DC / len
    test_loss = test_loss / len
    acc = acc/len
    TP = TP/len
    TN = TN/len
    FP = FP/len
    FN = FN/len
    # Putting the model back to training mode.
    print('DC:',DC,',test_loss:',test_loss,'accuracy:',acc,'TP:',TP,'TN:',TN,'FP:',FP,"FN:",FN)
    model.train()
    return DC,test_loss

test(testloader,testlen,0.5)



