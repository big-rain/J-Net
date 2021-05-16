import os
import cv2
import torchvision

from torch.utils.data import Dataset
from torchvision.utils import save_image

class Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        # 语义分割需要的图片的图片和标签
        self.name1 = os.listdir(os.path.join(path, "images"))
        self.name2 = os.listdir(os.path.join(path, "1st_manual"))
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.name1)
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

    def __getitem__(self, index):
        name1 = self.name1[index]
        # print(name1)
        name2 = self.name2[index]
        # print(name2)
        img_path = [os.path.join(self.path, i) for i in ("images", "1st_manual")]
        # print(os.path.join(img_path[0], name1))
        img_o = cv2.imread(os.path.join(img_path[0], name1))
        img_l = cv2.imread(os.path.join(img_path[1], name2))

        # _, img_l = cv2.VideoCapture(os.path.join(img_path[1], name2)).read()

        img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)

        img_o = self.__trans__(img_o, 256)
        img_l = self.__trans__(img_l, 256)
        return self.trans(img_o), self.trans(img_l)
if __name__ == '__main__':
    i = 1
    # 路径改一下
    dataset = Dataset(r"../data/DRIVE/training/")
    for a, b in dataset:
        print(i)
        # print(a.shape)
        # print(b.shape)
        save_image(a, f"./img/{i}.jpg", nrow=1)
        save_image(b, f"./img/{i}.png", nrow=1)
        i += 1
        if i > 2:
            break