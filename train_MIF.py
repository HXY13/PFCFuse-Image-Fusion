# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
# from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, EVCBlock
# from utils.dataset_MIF1 import H5Dataset
# from utils.dataset_MIF2 import H5Dataset
# from utils.dataset_MIF3 import H5Dataset
from utils.dataset_MIF4 import H5Dataset
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc, RMI_loss
import kornia


'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()
model_str = 'CDDFuse'

# . Set the hyper-parameters for training
num_epochs = 120 # total epoch
epoch_gap = 40  # epoches of Phase I 

lr = 1e-4
weight_decay = 0
batch_size = 8
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1. # alpha1
coeff_mse_loss_IF = 1.
coeff_rmi_loss_VF = 1.
coeff_rmi_loss_IF = 1.
coeff_decomp = 2.      # alpha2 and alpha4
coeff_tv = 5.

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5


# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
# BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
# DetailFuseLayer = nn.DataParallel(EVCBlock(64,64)).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()  
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIM(11, reduction='mean')


# data loader
# trainloader = DataLoader(H5Dataset(r"data/MSRS_trainMIF1_new_imgsize_128_stride_200.h5"),
# trainloader = DataLoader(H5Dataset(r"data/MSRS_trainMIF2_new_imgsize_128_stride_200.h5"),
# trainloader = DataLoader(H5Dataset(r"data/MSRS_trainMIF3_new_imgsize_128_stride_200.h5"),
trainloader = DataLoader(H5Dataset(r"data/MSRS_trainMIF4_new_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train_MIF': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    ''' train '''
    # for i, (data_MRI_1, data_CT) in enumerate(loader['train_MIF']):
    # for i, (data_MRI_2, data_PET) in enumerate(loader['train_MIF']):
    # for i, (data_MRI_3, data_SPECT) in enumerate(loader['train_MIF']):
    for i, (data_MRI, data_Img) in enumerate(loader['train_MIF']):
    #     data_MRI_1, data_CT = data_MRI_1.cuda(), data_CT.cuda()
    #     data_MRI_2, data_PET = data_MRI_2.cuda(), data_PET.cuda()
    #     data_MRI_3, data_SPECT = data_MRI_3.cuda(), data_SPECT.cuda()
        data_MRI, data_Img = data_MRI.cuda(), data_Img.cuda()

        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        if epoch < epoch_gap: #Phase I
            # feature_V_B, feature_V_D, _ = DIDF_Encoder(data_MRI_1)
            # feature_I_B, feature_I_D, _ = DIDF_Encoder(data_CT)

            # feature_V_B, feature_V_D, _ = DIDF_Encoder(data_MRI_2)
            # feature_I_B, feature_I_D, _ = DIDF_Encoder(data_PET)

            # feature_V_B, feature_V_D, _ = DIDF_Encoder(data_MRI_3)
            # feature_I_B, feature_I_D, _ = DIDF_Encoder(data_SPECT)

            feature_V_B, feature_V_D, _ = DIDF_Encoder(data_MRI)
            feature_I_B, feature_I_D, _ = DIDF_Encoder(data_Img)

            # data_MRI_hat, _ = DIDF_Decoder(data_MRI_1, feature_V_B, feature_V_D)
            # data_CT_hat, _ = DIDF_Decoder(data_CT, feature_I_B, feature_I_D)

            # data_MRI_hat, _ = DIDF_Decoder(data_MRI_2, feature_V_B, feature_V_D)
            # data_PET_hat, _ = DIDF_Decoder(data_PET, feature_I_B, feature_I_D)
            #
            # data_MRI_hat, _ = DIDF_Decoder(data_MRI_3, feature_V_B, feature_V_D)
            # data_SPECT_hat, _ = DIDF_Decoder(data_SPECT, feature_I_B, feature_I_D)

            data_MRI_hat, _ = DIDF_Decoder(data_MRI, feature_V_B, feature_V_D)
            data_Img_hat, _ = DIDF_Decoder(data_Img, feature_I_B, feature_I_D)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            # mse_loss_V = 5 * Loss_ssim(data_MRI_1, data_MRI_hat) + MSELoss(data_MRI_1, data_MRI_hat)
            # mse_loss_I = 5 * Loss_ssim(data_CT, data_CT_hat) + MSELoss(data_CT, data_CT_hat)

            # mse_loss_V = 5 * Loss_ssim(data_MRI_2, data_MRI_hat) + MSELoss(data_MRI_2, data_MRI_hat)
            # mse_loss_I = 5 * Loss_ssim(data_PET, data_PET_hat) + MSELoss(data_PET, data_PET_hat)

            # mse_loss_V = 5 * Loss_ssim(data_MRI_3, data_MRI_hat) + MSELoss(data_MRI_3, data_MRI_hat)
            # mse_loss_I = 5 * Loss_ssim(data_SPECT, data_SPECT_hat) + MSELoss(data_SPECT, data_SPECT_hat)

            mse_loss_V = 5 * Loss_ssim(data_MRI, data_MRI_hat) + MSELoss(data_MRI, data_MRI_hat)
            mse_loss_I = 5 * Loss_ssim(data_Img, data_Img_hat) + MSELoss(data_Img, data_Img_hat)

            # Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_MRI_1),

            # Gradient_loss=L1Loss(kornia.filters.SpatialGradient()(data_MRI_2),
            #
            # Gradient_loss=L1Loss(kornia.filters.SpatialGradient()(data_MRI_3),

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_MRI),
                                   kornia.filters.SpatialGradient()(data_MRI_hat))

            loss_decomp =  (cc_loss_D) ** 2/ (1.01 + cc_loss_B)

            # loss_dice_M = dice_coeff(data_MRI, data_MRI_hat)
            # loss_dice_I = dice_coeff(data_Img, data_Img_hat)

            loss_rmi_m = RMI_loss(data_MRI, data_MRI_hat)
            loss_rmi_i = RMI_loss(data_Img, data_Img_hat)

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss +\
                   coeff_rmi_loss_IF * loss_rmi_i + coeff_rmi_loss_VF * loss_rmi_m

            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()  
            optimizer2.step()
        else:  #Phase II
            # feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_MRI_1)
            # feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_CT)

            # feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_MRI_2)
            # feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_PET)

            # feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_MRI_3)
            # feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_SPECT)

            feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_MRI)
            feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_Img)

            feature_F_B = BaseFuseLayer(feature_I_B+feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D+feature_V_D)
            # data_Fuse, feature_F = DIDF_Decoder(data_MRI_1, feature_F_B, feature_F_D)
            # data_Fuse, feature_F = DIDF_Decoder(data_MRI_2, feature_F_B, feature_F_D)
            # data_Fuse, feature_F = DIDF_Decoder(data_MRI_3, feature_F_B, feature_F_D)
            data_Fuse, feature_F = DIDF_Decoder(data_MRI, feature_F_B, feature_F_D)

            # mse_loss_V = 5*Loss_ssim(data_MRI_1, data_Fuse) + MSELoss(data_MRI_1, data_Fuse)
            # mse_loss_I = 5*Loss_ssim(data_CT,  data_Fuse) + MSELoss(data_CT,  data_Fuse)

            # mse_loss_V = 5 * Loss_ssim(data_MRI_2, data_Fuse) + MSELoss(data_MRI_2, data_Fuse)
            # mse_loss_I = 5 * Loss_ssim(data_PET, data_Fuse) + MSELoss(data_PET, data_Fuse)
            #
            # mse_loss_V = 5 * Loss_ssim(data_MRI_3, data_Fuse) + MSELoss(data_MRI_3, data_Fuse)
            # mse_loss_I = 5 * Loss_ssim(data_SPECT, data_Fuse) + MSELoss(data_SPECT, data_Fuse)

            mse_loss_V = 5 * Loss_ssim(data_MRI, data_Fuse) + MSELoss(data_MRI, data_Fuse)
            mse_loss_I = 5 * Loss_ssim(data_Img, data_Fuse) + MSELoss(data_Img, data_Fuse)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            loss_decomp =   (cc_loss_D) ** 2 / (1.01 + cc_loss_B)  
            # fusionloss, _,_  = criteria_fusion(data_MRI_1, data_CT, data_Fuse)

            # fusionloss, _,_  = criteria_fusion(data_MRI_2, data_PET, data_Fuse)

            # fusionloss, _,_  = criteria_fusion(data_MRI_3, data_SPECT, data_Fuse)

            fusionloss, _, _ = criteria_fusion(data_MRI, data_Img, data_Fuse)

            loss = fusionloss + coeff_decomp * loss_decomp
            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()  
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

        # Determine approximate time left
        batches_done = epoch * len(loader['train_MIF']) + i
        batches_left = num_epochs * len(loader['train_MIF']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train_MIF']),
                loss.item(),
                time_left,
            )
        )

    # adjust the learning rate

    scheduler1.step()  
    scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6
    
if True:
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    # torch.save(checkpoint, os.path.join("models/CDDFuse_MIF_1"+timestamp+'.pth'))
    # torch.save(checkpoint, os.path.join("models/CDDFuse_MIF_2"+timestamp+'.pth'))
    # torch.save(checkpoint, os.path.join("models/CDDFuse_MIF_3"+timestamp+'.pth'))
    torch.save(checkpoint, os.path.join("models/CDDFuse_MIF_attpool" + timestamp + '.pth'))



