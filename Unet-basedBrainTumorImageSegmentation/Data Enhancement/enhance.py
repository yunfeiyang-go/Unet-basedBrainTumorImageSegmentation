import cv2
import os
import numpy as np

# 加载所有MRI图像和mask
#img_dir = './3-BrainTumor/images/'
#mask_dir = './3-BrainTumor/masks/'
img_dir = '/home/ug2020/ug520111910171/bioalgorithm/Brain/3-BrainTumor/images/'
mask_dir = '/home/ug2020/ug520111910171/bioalgorithm/Brain/3-BrainTumor/masks/'
image_files = os.listdir(img_dir)
mask_files = os.listdir(mask_dir)


# 定义数据增强操作
def augment(image, mask):

    # 随机翻转
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    # 随机旋转和缩放
    angle = np.random.randint(-10, 10)
    scale = np.random.uniform(0.8, 1.2)
    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, scale)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
    
    # 随机裁剪和填充
    x_offset, y_offset = np.random.randint(-50, 50, size=2)
    if x_offset > 0:
        image = np.pad(image[:, :-x_offset], ((0, 0), (x_offset, 0)), mode='constant')
        mask = np.pad(mask[:, :-x_offset], ((0, 0), (x_offset, 0)), mode='constant')
    else:
        image = np.pad(image[:, -x_offset:], ((0, 0), (0, -x_offset)), mode='constant')
        mask = np.pad(mask[:, -x_offset:], ((0, 0), (0, -x_offset)), mode='constant')
        image = np.roll(image, x_offset, axis=1)
        mask = np.roll(mask, x_offset, axis=1)

    if y_offset > 0:
        image = np.pad(image[:-y_offset, :], ((y_offset, 0), (0, 0)), mode='constant')
        mask = np.pad(mask[:-y_offset, :], ((y_offset, 0), (0, 0)), mode='constant')
    else:
        image = np.pad(image[-y_offset:, :], ((0, -y_offset), (0, 0)), mode='constant')
        mask = np.pad(mask[-y_offset:, :], ((0, -y_offset), (0, 0)), mode='constant')
        image = np.roll(image, y_offset, axis=0)
        mask = np.roll(mask, y_offset, axis=0)


    return image, mask

# 将增强后的图像和掩码保存到新的文件夹中
new_images_folder = '/home/ug2020/ug520111910171/bioalgorithm/Brain/3-BrainTumor/enhance/augmented_images/'
new_masks_folder = '/home/ug2020/ug520111910171/bioalgorithm/Brain/3-BrainTumor/enhance/augmented_masks/'
if not os.path.exists(new_images_folder):
    os.makedirs(new_images_folder)
if not os.path.exists(new_masks_folder):
    os.makedirs(new_masks_folder)

# 对每张MRI图像和mask进行数据增强
for i in range(len(image_files)):
    img_path = os.path.join(img_dir, image_files[i])
    mask_path = os.path.join(mask_dir, mask_files[i])
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to read image at {img_path}")

    image_aug_sum = img
    mask_aug_sum = mask

    # 进行3次数据增强，并将增强后的图像保存到新的文件夹中
    count = 0
    for j in range(3):
        image_aug, mask_aug = augment(img, mask)
        new_img_path = os.path.join(new_images_folder, image_files[i][:-4] + '_aug_' + str(count) + '.png')  #去掉.png
        new_mask_path = os.path.join(new_masks_folder, mask_files[i][:-4] + '_aug_' + str(count) + '.png')
        cv2.imwrite(new_img_path, image_aug)
        cv2.imwrite(new_mask_path, mask_aug)
        count += 1

print('Data augmentation finished!')
