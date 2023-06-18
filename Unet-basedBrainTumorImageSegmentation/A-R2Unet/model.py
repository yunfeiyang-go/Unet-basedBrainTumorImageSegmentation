import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from network import R2AttU_Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


model = R2AttU_Net(img_ch=1, output_ch=1).to(device)

class BrainTumorDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_filenames = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.img_filenames[idx])
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        img = self.transform(img)
        mask = self.transform(mask)

        return img.to(device), mask.to(device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


img_dir = './3-Brain-Tumor/enhance/augmented_images/'
mask_dir = './3-Brain-Tumor/enhance/augmented_masks/'

dataset = BrainTumorDataset(img_dir, mask_dir, transform=transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42))

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def calculate_pixel_accuracy(outputs, masks):
    predicted_labels = (outputs > 0.5).float()
    masks = masks == torch.max(masks)
    correct = (predicted_labels == masks).float().sum()
    total = masks.numel()
    accuracy = correct / total
    return accuracy


def calculate_dice_score(outputs, masks):
    smooth = 1e-6
    predicted_labels = (outputs > 0.5).float()
    intersection = (predicted_labels * masks).sum()
    union = predicted_labels.sum() + masks.sum()
    dice_score = (2 * intersection + smooth) / (union + smooth)
    return dice_score


def calculate_metrics(outputs, masks):
    predictions = (outputs > 0.5).float()
    masks = masks == torch.max(masks)
    TP = ((predictions == 1) & (masks == 1)).sum().item()
    TN = ((predictions == 0) & (masks == 0)).sum().item()
    FP = ((predictions == 1) & (masks == 0)).sum().item()
    FN = ((predictions == 0) & (masks == 1)).sum().item()
    return TP, TN, FP, FN


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    train_accuracy = 0
    train_dice_score = 0
    epoch_TP = epoch_TN = epoch_FP = epoch_FN = 0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_accuracy += calculate_pixel_accuracy(outputs, masks)
        train_dice_score += calculate_dice_score(outputs, masks)

        TP, TN, FP, FN = calculate_metrics(outputs, masks)
        epoch_TP += TP
        epoch_TN += TN
        epoch_FP += FP
        epoch_FN += FN

    precision = epoch_TP / (epoch_TP + epoch_FP + 1e-6)
    recall = epoch_TP / (epoch_TP + epoch_FN + 1e-6)
    specificity = epoch_TN / (epoch_TN + epoch_FP + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    print("Train:")
    print(f"TP: {epoch_TP}, TN: {epoch_TN}, FN: {epoch_FN}, FP: {epoch_FP}")
    print(f"Train Precision: {precision}, Recall: {recall}, Specificity:{specificity}, F1-score:{f1_score}")
    return train_loss / len(train_loader.dataset), train_accuracy / len(train_loader), train_dice_score / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_accuracy = 0
    val_dice_score = 0
    epoch_TP = epoch_TN = epoch_FP = epoch_FN = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
            val_accuracy += calculate_pixel_accuracy(outputs, masks)
            val_dice_score += calculate_dice_score(outputs, masks)

            TP, TN, FP, FN = calculate_metrics(outputs, masks)
            epoch_TP += TP
            epoch_TN += TN
            epoch_FP += FP
            epoch_FN += FN

        precision = epoch_TP / (epoch_TP + epoch_FP + 1e-6)
        recall = epoch_TP / (epoch_TP + epoch_FN + 1e-6)
        specificity = epoch_TN / (epoch_TN + epoch_FP + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        print("Validation:")
        print(f"TP: {epoch_TP}, TN: {epoch_TN}, FN: {epoch_FN}, FP: {epoch_FP}")
        print(f"Validation Precision: {precision}, Recall: {recall}, Specificity:{specificity}, F1-score:{f1_score}")

    return val_loss / len(val_loader.dataset), val_accuracy / len(val_loader), val_dice_score / len(val_loader)


num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss, train_accuracy, train_dice_score = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy, val_dice_score = validate(model, val_loader, criterion, device)
    print(f"Train Loss: {train_loss:.4f}, Train pixel accuracy: {train_accuracy:.4f}, Train Dice Score: {train_dice_score:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val pixel accuracy: {val_accuracy:.4f}, Val Dice Score: {val_dice_score:.4f}")
    print("==============================================")
torch.save(model.state_dict(), './model_final.pth')



def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_accuracy = 0
    test_dice_score = 0
    epoch_TP = epoch_TN = epoch_FP = epoch_FN = 0
    predictions = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            predicted_masks = outputs > 0.5
            predictions.extend(predicted_masks.cpu().numpy())

            masks = torch.sigmoid(masks)
            loss = criterion(outputs, masks)
            test_loss += loss.item() * images.size(0)
            test_accuracy += calculate_pixel_accuracy(outputs, masks)
            test_dice_score += calculate_dice_score(outputs, masks)

            TP, TN, FP, FN = calculate_metrics(outputs, masks)
            epoch_TP += TP
            epoch_TN += TN
            epoch_FP += FP
            epoch_FN += FN

        precision = epoch_TP / (epoch_TP + epoch_FP + 1e-6)
        recall = epoch_TP / (epoch_TP + epoch_FN + 1e-6)
        specificity = epoch_TN / (epoch_TN + epoch_FP + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        print("Testing:")
        print(f"TP: {epoch_TP}, TN: {epoch_TN}, FN: {epoch_FN}, FP: {epoch_FP}")
        print(f"Testing Precision: {precision}, Recall: {recall}, Specificity:{specificity}, F1-score:{f1_score}")
        test_loss /= len(test_loader.dataset)
        test_accuracy /= len(test_loader.dataset)
        test_dice_score /= len(test_loader.dataset)
        print(f"Test Loss: {test_loss:.4f}, Test pixel accuracy: {test_accuracy:.4f}, Test Dice Score: {test_dice_score:.4f}")
    return predictions


model.load_state_dict(torch.load('./model_final.pth'))
model.eval()
predictions = test(model, test_loader, criterion, device)
