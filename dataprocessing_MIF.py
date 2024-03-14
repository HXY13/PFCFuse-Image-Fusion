import os
import h5py
import numpy as np
from tqdm import tqdm
from skimage.io import imread
# 加-start
from PIL import Image
# 加-end

def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist


def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """Determine if an image is low contrast."""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold

# 加-start
def is_grayscale_image(image_path):
    '''
    return True 为灰度图
    False 不是灰度图
    '''
    image= Image.open(image_path)
    if image.mode == 'L':
        return True
    else:
        return False
# 加-end

data_name = "MSRS_train"
img_size = 128  # patch size
# stride = 200  # patch stride
stride = 64  # patch stride

# CT_files = sorted(get_img_file(r"MSRS_train/harvard/train/CT-MRI/CT"))
# MRI_1_files = sorted(get_img_file(r"MSRS_train/harvard/train/CT-MRI/MRI"))
#
# PET_files = sorted(get_img_file(r"MSRS_train/harvard/train/PET-MRI/PET"))
# MRI_2_files = sorted(get_img_file(r"MSRS_train/harvard/train/PET-MRI/MRI"))
#
# SPECT_files = sorted(get_img_file(r"MSRS_train/harvard/train/SPECT-MRI/SPECT"))
# MRI_3_files = sorted(get_img_file(r"MSRS_train/harvard/train/SPECT-MRI/MRI"))
# try-all
img_a_files = sorted(get_img_file(r"MSRS_train/harvard/train/CT-PET-SPECT"))
MRI_a_files = sorted(get_img_file(r"MSRS_train/harvard/train/MRI"))



# assert len(CT_files) == len(MRI_1_files)
# assert len(PET_files) == len(MRI_2_files)
# assert len(SPECT_files) == len(MRI_3_files)
# try-all
assert len(img_a_files) == len(MRI_a_files)

h5f = h5py.File(os.path.join('./data',
                             data_name + 'MIF4_new_imgsize_' + str(img_size) + "_stride_" + str(stride) + '.h5'),
                'w')
# h5_ct = h5f.create_group('ct_patchs')
# h5_mri1 = h5f.create_group('mri1_patchs')

# h5_pet = h5f.create_group('pet_patchs')
# h5_mri2 = h5f.create_group('mri2_patchs')

# h5_spect = h5f.create_group('spect_patchs')
# h5_mri3 = h5f.create_group('mri3_patchs')

# try-all
h5_img = h5f.create_group('img_patchs')
h5_mri_a = h5f.create_group('mria_patchs')


train_num = 0
# # CT_MRI
# for i in tqdm(range(len(CT_files))):
#     I_CT = imread(CT_files[i]).astype(np.float32)[None, :, :] / 255.  # [1, H, W] Float32
#     I_MRI_1 = imread(MRI_1_files[i]).astype(np.float32)[None, :, :] / 255.  # [1, H, W] Float32
#
#     # crop
#     I_CT_Patch_Group = Im2Patch(I_CT, img_size, stride)
#     I_MRI_1_Patch_Group = Im2Patch(I_MRI_1, img_size, stride)
#
#     for ii in range(I_CT_Patch_Group.shape[-1]):
#         bad_CT = is_low_contrast(I_CT_Patch_Group[0, :, :, ii])
#         bad_MRI_1 = is_low_contrast(I_MRI_1_Patch_Group[0, :, :, ii])
#         # Determine if the contrast is low
#         if not (bad_CT or bad_MRI_1):
#             avl_CT = I_CT_Patch_Group[0, :, :, ii]  # available CT
#             avl_MRI_1 = I_MRI_1_Patch_Group[0, :, :, ii]
#             avl_CT = avl_CT[None, ...]
#             avl_MRI_1 = avl_MRI_1[None, ...]
#
#             h5_ct.create_dataset(str(train_num), data=avl_CT,
#                                  dtype=avl_CT.dtype, shape=avl_CT.shape)
#             h5_mri1.create_dataset(str(train_num), data=avl_MRI_1,
#                                   dtype=avl_MRI_1.dtype, shape=avl_MRI_1.shape)
#             train_num += 1

# # PET_MRI
# for i in tqdm(range(len(PET_files))):
# #     I_PET = imread(PET_files[i]).astype(np.float32)[None, :, :] / 255.  # [1, H, W] Float32
#     I_PET = imread(PET_files[i]).astype(np.float32).transpose(2,0,1)/255. # [3, H, W] Uint8->float32
#     I_PET = rgb2y(I_PET) # [1, H, W] Float32
#     I_MRI_2 = imread(MRI_2_files[i]).astype(np.float32)[None, :, :] / 255.  # [1, H, W] Float32
#
#     # crop
#     I_PET_Patch_Group = Im2Patch(I_PET, img_size, stride)
#     I_MRI_2_Patch_Group = Im2Patch(I_MRI_2, img_size, stride)
#
#     for ii in range(I_PET_Patch_Group.shape[-1]):
#         bad_PET = is_low_contrast(I_PET_Patch_Group[0, :, :, ii])
#         bad_MRI_2 = is_low_contrast(I_MRI_2_Patch_Group[0, :, :, ii])
#         # Determine if the contrast is low
#         if not (bad_PET or bad_MRI_2):
#             avl_PET = I_PET_Patch_Group[0, :, :, ii]  # available CT
#             avl_MRI_2 = I_MRI_2_Patch_Group[0, :, :, ii]
#             avl_PET = avl_PET[None, ...]
#             avl_MRI_2 = avl_MRI_2[None, ...]
#
#             h5_pet.create_dataset(str(train_num), data=avl_PET,
#                                  dtype=avl_PET.dtype, shape=avl_PET.shape)
#             h5_mri2.create_dataset(str(train_num), data=avl_MRI_2,
#                                   dtype=avl_MRI_2.dtype, shape=avl_MRI_2.shape)
#             train_num += 1

# # SPECT_MRI
# for i in tqdm(range(len(SPECT_files))):
# #     I_SPECT = imread(SPECT_files[i]).astype(np.float32)[None, :, :] / 255.  # [1, H, W] Float32
#     I_SPECT = imread(SPECT_files[i]).astype(np.float32).transpose(2,0,1)/255. # [3, H, W] Uint8->float32
#     I_SPECT = rgb2y(I_SPECT) # [1, H, W] Float32
#     I_MRI_3 = imread(MRI_3_files[i]).astype(np.float32)[None, :, :] / 255.  # [1, H, W] Float32
#
#     # crop
#     I_SPECT_Patch_Group = Im2Patch(I_SPECT, img_size, stride)
#     I_MRI_3_Patch_Group = Im2Patch(I_MRI_3, img_size, stride)
#
#     for ii in range(I_SPECT_Patch_Group.shape[-1]):
#         bad_SPECT = is_low_contrast(I_SPECT_Patch_Group[0, :, :, ii])
#         bad_MRI_3 = is_low_contrast(I_MRI_3_Patch_Group[0, :, :, ii])
#         # Determine if the contrast is low
#         if not (bad_SPECT or bad_MRI_3):
#             avl_SPECT = I_SPECT_Patch_Group[0, :, :, ii]  # available SPECT
#             avl_MRI_3 = I_MRI_3_Patch_Group[0, :, :, ii]
#             avl_SPECT = avl_SPECT[None, ...]
#             avl_MRI_3 = avl_MRI_3[None, ...]
#
#             h5_spect.create_dataset(str(train_num), data=avl_SPECT,
#                                  dtype=avl_SPECT.dtype, shape=avl_SPECT.shape)
#             h5_mri3.create_dataset(str(train_num), data=avl_MRI_3,
#                                   dtype=avl_MRI_3.dtype, shape=avl_MRI_3.shape)
#             train_num += 1

# try-all
for i in tqdm(range(len(img_a_files))):
    # print(is_grayscale_image(img_a_files[i]))
    if(is_grayscale_image(img_a_files[i])==True):
        I_img = imread(img_a_files[i]).astype(np.float32)[None, :, :] / 255.  # [1, H, W] Float32
    else:
        I_img = imread(img_a_files[i]).astype(np.float32).transpose(2,0,1)/255. # [3, H, W] Uint8->float32
        I_img = rgb2y(I_img) # [1, H, W] Float32

    # I_img = imread(img_a_files[i]).astype(np.float32)[None, :, :] / 255.  # [1, H, W] Float32
    I_MRI_a = imread(MRI_a_files[i]).astype(np.float32)[None, :, :] / 255.  # [1, H, W] Float32

    # crop
    I_img_Patch_Group = Im2Patch(I_img, img_size, stride)
    I_MRI_a_Patch_Group = Im2Patch(I_MRI_a, img_size, stride)

    for ii in range(I_img_Patch_Group.shape[-1]):
        bad_img = is_low_contrast(I_img_Patch_Group[0, :, :, ii])
        bad_MRI_a = is_low_contrast(I_MRI_a_Patch_Group[0, :, :, ii])
        # Determine if the contrast is low
        if not (bad_img or bad_MRI_a):
            avl_img = I_img_Patch_Group[0, :, :, ii]  # available CT
            avl_MRI_a = I_MRI_a_Patch_Group[0, :, :, ii]
            avl_img = avl_img[None, ...]
            avl_MRI_a = avl_MRI_a[None, ...]

            h5_img.create_dataset(str(train_num), data=avl_img,
                                 dtype=avl_img.dtype, shape=avl_img.shape)
            h5_mri_a.create_dataset(str(train_num), data=avl_MRI_a,
                                  dtype=avl_MRI_a.dtype, shape=avl_MRI_a.shape)
            train_num += 1

h5f.close()

with h5py.File(os.path.join('data',
                            data_name + 'MIF4_new' + '_imgsize_' + str(img_size) + "_stride_" + str(stride) + '.h5'),
               "r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name)







