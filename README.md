# Unet-basedBrainTumorImageSegmentation
BrainTumor Image Segmentation Using U-Net and Improved Algorithm

Image segmentation is the process of dividing an image into multiple sub-regions for better understanding, processing and analysis. In fields such as medical imaging, image segmentation is widely used for diagnosis and treatment. Through this optimized processing, doctors can provide more detailed diagnostic results and better guide treatment procedures, improving treatment effectiveness.

Magnetic resonance imaging (MRI), which provides information about the shape, size, location, and metabolism of brain tumors, is a non-invasive imaging technique that can enhance contrast between different brain tissues and generate images of different modalities, thereby highlighting different tissues and making it easier for doctors to compare and obtain useful pathological information.

We selected U-Net as the basic image segmentation algorithm and compared it with three improved algorithms, R2U-Net, Attention-Unet, and Attention-R2Unet, trying to find the best-performing model and the most suitable hyperparameters in this field.

In our code repository, there are codes for data augmentation, building four models, and auxiliary codes for outputting predicted images and submitting tasks.

In our model evaluation, we did not use the conventional TN, TP, FN, FP calculation method. Because we focus on the tumor and the part predicted as a tumor, we do not need to treat or pay attention to large areas of black background and normal tissue. Therefore, the range involved in calculating accuracy is the union of the two aforementioned. By completely excluding the TN part, we can amplify the difference signal between models and facilitate us to pay attention to a large number of clinically dangerous FP signal.

If you want to use our code, please note that the python version should be 3.6 and a GPU is recommended.
