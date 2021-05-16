import cv2
import os
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataloaders.loader_2 import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

from model.resnet_aspp_v1 import L_net


# 计算DICE系数，即DSI
def calDSI(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    DSI_s, DSI_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                DSI_s += 1
            if binary_GT[i][j] == 255:
                DSI_t += 1
            if binary_R[i][j] == 255:
                DSI_t += 1
    DSI = 2 * DSI_s / (DSI_t + 1e-5)
    # print(DSI)
    return DSI


# 计算VOE系数，即VOE
def calVOE(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    VOE_s, VOE_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                VOE_s += 1
            if binary_R[i][j] == 255:
                VOE_t += 1
    VOE = 2 * (VOE_t - VOE_s) / (VOE_t + VOE_s)
    return VOE


# 计算RVD系数，即RVD
def calRVD(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    RVD_s, RVD_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                RVD_s += 1
            if binary_R[i][j] == 255:
                RVD_t += 1
    RVD = RVD_t / RVD_s - 1
    return RVD


# 计算Prevision系数，即Precison
def calPrecision(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    P_s, P_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                P_s += 1
            if binary_R[i][j] == 255:
                P_t += 1

    Precision = P_s / P_t
    return Precision


# 计算Recall系数，即Recall
def calRecall(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    R_s, R_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                R_s += 1
            if binary_GT[i][j] == 255:
                R_t += 1

    Recall = R_s / R_t
    return Recall




if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = L_net(3, 1).to(device)
    # path = '../data/liver/val/'
    # model.load_state_dict(torch.load(path))
    #
    # loader = DataLoader(Dataset(path), batch_size=4, shuffle=True, num_workers=4)



    dice = 0.0
    recall = 0.0
    Precision = 0.0
    # step 1：读入图像，并灰度化
    #
    pre_path = '../res_l_dice/p1'
    label_path = '../res_l_dice/l1'

    # pre_path = '../res_uNet/predic'
    # label_path = '../res_uNet/label'

    # pre_path = '../res_deeplab/predic'
    # label_path = '../res_deeplab/label'

    # pre_path = '../res_Fcn/predictied'
    # label_path = '../res_Fcn/label'

    pre_imgs = os.listdir(pre_path)
    lab_imgs = os.listdir(label_path)

    for i, p in enumerate(pre_imgs):
        # print(os.path.join(pre_path, p))
        # print(os.path.join(label_path, pre_imgs[i]))
        img_GT = cv2.imread(os.path.join(pre_path, p), 0)
        img_R = cv2.imread(os.path.join(label_path, pre_imgs[i]), 0)
    # imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   # 灰度化
    # img_GT = img_GT[:,:,[2, 1, 0]]
    # img_R  = img_R[:,: [2, 1, 0]]


    # step2：二值化
    # 利用大律法,全局自适应阈值 参数0可改为任意数字但不起作用
        ret_GT, binary_GT = cv2.threshold(img_GT, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ret_R, binary_R = cv2.threshold(img_R, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # step 3： 显示二值化后的分割图像与真值图像
    # plt.figure()
    # plt.subplot(121), plt.imshow(binary_GT), plt.title('真值图')
    # plt.axis('off')
    # plt.subplot(122), plt.imshow(binary_R), plt   .title('分割图')
    # plt.axis('off')
    # plt.show()

    # step 4：计算DSI
        dice += calDSI(binary_GT, binary_R)
        Precision += calPrecision(binary_GT, binary_R)
        recall += calRecall(binary_GT, binary_R)
    print('（1）DICE计算结果，DSI = {0:.4}'.format(dice / 30))  # 保留四位有效数字
    #
    # # step 5：计算VOE
    #     print('（2）VOE计算结果，VOE = {0:.4}'.format(calVOE(binary_GT, binary_R)))
    #
    # # step 6：计算RVD
    #     print('（3）RVD计算结果, RVD = {0:.4}'.format(calRVD(binary_GT, binary_R)))

    # step 7：计算Precision
    print('（4）Precision计算结果, Precision = {0:.4}'.format(Precision/30))

    # step 8：计算Recall
    print('（5）Recall计算结果，Recall = {0:.4}'.format(recall/30))
