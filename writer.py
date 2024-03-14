import numpy as np
from PIL import Image
from torch.utils.tensorboard import  SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")

# writer.add_image()
# y = x

image_path = "test_img/MRI_PET/PET/11.png"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

print(type(img_array))
print(img_array.shape)

writer.add_image("train", img_array, 2, dataformats='HWC')
writer.add_image("test", img_array, 1, dataformats='HWC')
# for i in range(100):
#     writer.add_scalar("y=x",i,i)

writer.close()

# 通过transforms.ToTensor去看两个问题
# 1. transforms该如何使用(python)
# 2. 为什么我们需要Tensor数据类型

img_path = "MSRS_train/harvard/train/CT-PET-SPECT/3021.png"
img = Image.open(img_path)

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

print(tensor_img)