import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        # 加
        loss_rmi_v=relative_diff_loss(image_vis, generate_img)
        loss_rmi_i=relative_diff_loss(image_ir, generate_img)
        x_rmi_max=torch.max(loss_rmi_v, loss_rmi_i)
        loss_rmi=F.l1_loss(x_rmi_max, generate_img)
        # 加
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        # loss_total=loss_in+10*loss_grad
        #改
        loss_total = loss_in + 10 * loss_grad + loss_rmi
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()


# def dice_coeff(img1, img2):
#     smooth = 1.
#     num = img1.size(0)
#     m1 = img1.view(num, -1)  # Flatten
#     m2 = img2.view(num, -1)  # Flatten
#     intersection = (m1 * m2).sum()
#
#     return 1 - (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

# 用来衡量图像之间的平均灰度差异
def relative_diff_loss(img1, img2):
    # 计算图像的平均灰度值
    mean_intensity_img1 = torch.mean(img1)
    mean_intensity_img2 = torch.mean(img2)
    # print("mean_intensity_img1")
    # print(mean_intensity_img1)
    # print("mean_intensity_img2")
    # print(mean_intensity_img2)

    # 计算relative_diff
    epsilon = 1e-10  # 防止除零错误
    relative_diff  = abs((mean_intensity_img1 - mean_intensity_img2) / (mean_intensity_img1 + epsilon))

    return relative_diff

# 互信息MI损失
# def mutual_information_loss(img1, img2):
#     # 计算 X 和 Y 的熵
#     entropy_img1 = -torch.mean(torch.sum(F.softmax(img1, dim=-1) * F.log_softmax(img1, dim=-1), dim=-1))
#     entropy_img2 = -torch.mean(torch.sum(F.softmax(img2, dim=-1) * F.log_softmax(img2, dim=-1), dim=-1))
#
#     # 计算 X 和 Y 的联合熵
#     joint_entropy = -torch.mean(torch.sum(F.softmax(img1, dim=-1) * F.log_softmax(img2, dim=-1), dim=-1))
#
#     # 计算互信息损失
#     mutual_information = entropy_img1 + entropy_img2 - joint_entropy
#
#     return mutual_information


# # 余弦相似度计算
# def cosine_similarity(img1, img2):
#     # Flatten the tensors
#     img1_flat = img1.view(-1)
#     img2_flat = img2.view(-1)
#
#     # Calculate cosine similarity
#     similarity = Fine_similarity(img1_flat, img2_flat, dim=0)
#
#     loss = torch.abs(similarity - 1)
#     return loss.item()  # Convert to Python float
