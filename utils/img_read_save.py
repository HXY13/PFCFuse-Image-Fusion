import numpy as np
import cv2
import os
from skimage.io import imsave

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    # img_BGR = cv2.imread(path)
    # print(img_BGR)
    # if img_BGR is not None:
    #     img_BGR = img_BGR.astype('float32')
    # else:
    #     print("处理图像加载失败的情况")
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Gray_pic
    imsave(os.path.join(savepath, "{}.png".format(imagename)),image.astype(np.uint8))

