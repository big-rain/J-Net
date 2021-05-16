from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2
from os.path import splitext
import torchvision
import os
from torchvision.utils import save_image

class Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        # 语义分割需要的图片的图片和标签
        self.imgs_dir = os.path.join(path, "image")
        self.masks_dir = os.path.join(path, "label")
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        self.ids = [splitext(file)[0] for file in listdir(self.imgs_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    def __trans__(self, img, size):
        h, w = img.shape[0:2]
        _w = _h = size
        scale = min(_h/h, _w/w)
        h = int(h*scale)
        w = int(w*scale)

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

        top = (_h - h) // 2
        left = (_w - w) // 2

        bottom = _h-h-top
        right = _w-w-left

        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return new_img
    @classmethod
    def preprocess(cls, pil_img, scale):
        width, height = pil_img.shape[:2][::-1]
        newW, newH = int(scale * width), int(scale * height)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans


    def __getitem__(self, index):
        idx = self.ids[index]

        mask_file = os.path.join(self.masks_dir,  f'{idx}_mask.png')
        # mask_file = os.path.join(self.masks_dir, f'{idx}.png')
        img_file = os.path.join(self.imgs_dir,  f'{idx}.png')
        img = cv2.imread(img_file)
        mask = cv2.imread(mask_file)
        # print(mask_file)
        # print(img.shape)
        # print(mask.shape)
        # mask = Image.open(mask_file)
        # img = Image.open(img_file)


        img_o = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # # img_l = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        img_l = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)


        img_o = self.__trans__(img_o, 256)
        img_l = self.__trans__(img_l, 256)

        # img_l = cv2.cvtColor(img_l, cv2.COLOR_RGB2GRAY)
        # img_l = np.expand_dims(img_l, axis=2)
        # mask = img_l.transpose((2, 0, 1))


        return self.trans(img_o), self.trans(img_l)
