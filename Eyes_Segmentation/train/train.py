import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model.u_net import U_net
from model.resnet_aspp_v1 import L_net
from model.deeplab import DeepLab
from model.FCN import FCN8s, VGGNet
from dataloaders.loader_2 import Dataset
# from dataloaders.liver_data import Dataset

from loss.dice_loss import dice_loss, dice_coeff
from function.PC import calDSI, calRVD, calVOE, calPrecision, calRecall


# def dice_coeff(pred, target):
#     smooth = 1.
#     num = pred.size(0)
#     m1 = pred.view(num, -1)  # Flatten
#     m2 = target.view(num, -1)  # Flatten
#     intersection = (m1 * m2).sum()
#
#     return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class trainer:
    def __init__(self, path, model, model_copy, img_save_path):
        self.path = path
        self.model = model
        self.model_copy = model_copy
        self.img_save_path = img_save_path

        # self.th = torch.nn.Sigmoid()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = DeepLab(backbone='resnet', output_stride=16).to(self.device)
        # self.net = L_net(in_channels=3, out_classes=3).to(self.device)
        # self.net = U_net(n_channels=3, n_classes=3).to(self.device)
        # self.vgg_model = VGGNet().to(self.device)
        # self.net = FCN8s(pretrained_net=self.vgg_model, n_class=3).to(self.device)

        # optimizae function
        initial_lr = 0.001
        self.opti = torch.optim.Adam(self.net.parameters(), lr=initial_lr)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opti, 10, eta_min=1e-5)

        # loss function
        self.loss = nn.BCELoss()
        # self.loss = dice_loss()

        self.loader = DataLoader(Dataset(path), batch_size=4, shuffle=True, num_workers=4)
        if os.path.exists(self.model):
            self.net.load_state_dict(torch.load(model))
            print(f"Loaded{model}!")
        else:
            print("No, Param!")
        os.makedirs(img_save_path, exist_ok=True)

    def train(self, stop_value):
        epoch = 1
        while True:
            for inputs, labels in tqdm(self.loader, desc=f"Epoch{epoch}/stop_value", ascii=True, total=len(self.loader)):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    out = self.net(inputs)

                    # print('input: ', inputs.shape)
                    # print('label: ', labels.shape)
                    # print('output:', out.shape)

                    loss = self.loss(out, labels)
                    # loss = dice_loss(out, labels)
                    self.opti.zero_grad()
                    loss.backward()
                    self.opti.step()
                    # self.scheduler.step()
                    DIS = dice_coeff(out[0], labels[0])
                    x = inputs[0]
                    x_ = out[0]
                    y = labels[0]


                    im = torch.stack([x, x_, y], 0)
                    # im = torch.stack([x_], 0)
                    save_image(im.cpu(), os.path.join(self.img_save_path, f"{epoch}.png"))

            print(f"\nEpoch: {epoch}/{stop_value}, Loss:{loss}, DIC{DIS}")
            torch.save(self.net.state_dict(), self.model)

            if epoch % 20 == 0:
                torch.save(self.net.state_dict(), self.model_copy.format(epoch, loss))
                print("model_copy is saved !")
            if epoch > stop_value:
                break
            epoch += 1

    '''
    def validation(self, epoch):
        self.model.eval()
        test_loss = 0.0
        for image, target in tqdm(self.val_loader, ascii=True, total=len(self.val_loader)):
            image, target = image.cuda(), target.cuda()
            # print(target.shape)
            with torch.no_grad():
                output = self.model(image)
            # print(output.shape)
            loss = dice_loss(output, target)
            test_loss += loss.item()

            self.opti.zero_grad()
            loss.backward()
            self.opti.step()

            x = image[0]
            x_ = output[0]
            y = target[0]

            im = torch.stack([x, x_, y], 0)
            save_image(im.cpu(), os.path.join(r'../res_2', f"{loss}.png"))
                # pred = output.data.cpu().numpy()
                # target = target.cpu().numpy()
                # pred = np.argmax(pred, axis=1)
                # target = np.argmax(target, axis=1)

        # step 4：计算DSI
        print('（1）DICE计算结果，DSI = {0:.4}'.format(dice_coeff(output, target)))  # 保留四位有效数字

        # step 7：计算Precision
        print('（4）Precision计算结果, Precision = {0:.4}'.format(calPrecision(target, output)))

        # step 8：计算Recall
        print('（5）Recall计算结果，Recall = {0:.4}'.format(calRecall(target, output)))
        '''

if __name__ == '__main__':
    path = 'model_300_0.0022984298411756754.plt'
    # model = U_net(n_channels=3, n_classes=3)
    # t = trainer(r"../data/liver/train/", path, r'./model_{}_{}.plt', img_save_path=r'../result_Lnet')
    t = trainer(r"../data/liver/train", r'./model.plt', r'./model_{}_{}.plt', img_save_path=r'../out_deep1')

    t.train(100)
    # for epoch in range(0, 300):
    #     t.train(epoch)
    #     if not epoch % 10 == 0:
    #         t.validation(epoch)