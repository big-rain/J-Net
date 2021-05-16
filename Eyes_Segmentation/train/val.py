import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from util.summaries import TensorboardSummary
from torchvision import transforms
from model.u_net import U_net
from model.resnet_aspp_v1 import L_net
from function.PC import *
from torchvision.utils import save_image
from dataloaders.loader_2 import Dataset
import logging
from model.FCN import VGGNet, FCN8s
from tqdm import tqdm
from torch.utils.data import DataLoader
# from loss.dice_loss import dice_coeff
from model.deeplab import DeepLab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class val():
    def __init__(self, model_pth, val_data_dir):
        self.model = L_net(in_channels=3, out_classes=3).to(device)
        # self.model = U_net(n_channels=3, n_classes=3).to(device)
        # self.model = model = DeepLab(backbone='resnet', output_stride=16).to(device)
        # self.vgg_model = VGGNet().to(device)
        # self.model = FCN8s(pretrained_net=self.vgg_model, n_class=3).to(device)

        self.model_pth = model_pth
        # print(self.model_pth)
        self.val_data_dir = val_data_dir
        self.liver_dataset = Dataset(self.val_data_dir)

        self.val_loader = DataLoader(self.liver_dataset, batch_size=1)
        self.criterion = nn.BCELoss()

        if os.path.exists(self.model_pth):
            self.model.load_state_dict(torch.load(self.model_pth))
            print(f"Loaded{model_pth}!")
        else:
            print("No, Param!")

    def validation(self):
        self.model.eval()
        test_loss = 0.0
        DIC = 0.0
        for image, target in tqdm(self.val_loader,  ascii=True, total=len(self.val_loader)):
            image, target = image.cuda(), target.cuda()
            # print(target.shape)
            with torch.no_grad():
                output = self.model(image)
            # print(output.shape)
            loss = self.criterion(output, target)
            test_loss += loss.item()

            binary_R = output[0]
            binary_GT = target[0]

            # binary_GT = torch.squeeze(binary_GT)
            # binary_R = torch.squeeze(binary_R)
            DIC += dice_coeff(binary_R, binary_GT)
            # print(DIC)
            im = torch.stack([binary_R], 0)
            im_mask = torch.stack([binary_GT])
            save_image(im.cpu(), os.path.join(r'../res_l_dice/50', f"{loss}.png"))
            save_image(im_mask.cpu(), os.path.join(r'../res_l_dice/50mask', f"{loss}.png"))
        print(DIC/30)
            # save_image(im.cpu(), os.path.join(r'../res_uNet/predic', f"{loss}.png"))
            # save_image(im_mask.cpu(), os.path.join(r'../res_uNet/label', f"{loss}.png"))
            # save_image(im.cpu(), os.path.join(r'../res_deeplab/predic', f"{loss}.png"))
            # save_image(im_mask.cpu(), os.path.join(r'../res_deeplab/label', f"{loss}.png"))
            # save_image(im.cpu(), os.path.join(r'../res_Fcn/predictied', f"{loss}.png"))
            # save_image(im_mask.cpu(), os.path.join(r'../res_Fcn/label', f"{loss}.png"))

            # print(binary_GT.shape)
            # print(binary_R.shape)


            # step 4：计算DSI
            # print('（1）DICE计算结果，DSI = {0:.4}'.format(calDSI(binary_GT, binary_R)))  # 保留四位有效数字

            # print('（1）DICE计算结果，DSI = {0:.4}'.format(dice_coeff(output, target)))

            # step 7：计算Precision

        #     dice += dice_coeff(binary_R, binary_GT)
        # print('（4）Precision计算结果, Precision = {0:.4}'.format(dice/20))

            # step 8：计算Recall
            # print('（3）recall计算结果，recall = {0:.4}'.format(recall(binary_R, binary_GT)))
            # print('***'*30)

if __name__ == "__main__":
    # path = 'model_300_0.0026035953778773546.plt'
    # path = 'fcn_checkpoint/model_250_0.0031534917652606964.plt'
    path = 'model_60_0.027279913425445557.plt'
    # path = 'deeplab_checkpoint/model_300_0.0025061173364520073.plt'
    # path = 'U_Net_checkpoint/model_250_0.002288644667714834.plt'
    trainer = val(path, r"../data/cell/test/")

    # for epoch in range(0, 20):
    trainer.validation()
