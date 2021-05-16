import os
import numpy as np
from function.score import Evaluator
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model.u_net import U_net
# from dataloaders.liver_data import Dataset
from dataloaders.loader_2 import Dataset

from model.resnet_aspp_v1 import L_net
class trainer:
    def __init__(self, path, model, model_copy, img_save_path):
        self.path = os.path.join(path, 'train')
        self.val = os.path.join(path, 'val')
        self.model = model
        self.model_copy = model_copy
        self.img_save_path = img_save_path

        self.th = torch.nn.Sigmoid()

        # Define Evaluator
        self.evaluator = Evaluator(2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = L_net(in_channels=3, out_classes=3).to(self.device)
        # self.net = U_net(n_channels=3, n_classes=3).to(self.device)

        # optimizae function
        self.opti = torch.optim.Adam(self.net.parameters())
        # loss function
        self.loss = nn.BCELoss()

        # dataloader
        self.train_loader = DataLoader(Dataset(self.path), batch_size=4, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(Dataset(self.val), batch_size=4, shuffle=True, num_workers=4)

        if os.path.exists(self.model):
            self.net.load_state_dict(torch.load(model))
            print(f"Loaded{model}!")
        else:
            print("No, Param!")
        os.makedirs(img_save_path, exist_ok=True)



    def train(self, stop_value):
        epoch = 1
        test_loss = 0.0
        while True:
            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch{epoch}/stop_value", ascii=True, total=len(self.train_loader)):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    out = self.net(inputs)
                    # out = self.th(out)
                    # print(out.shape)
                    loss = self.loss(out, labels)


                    self.opti.zero_grad()
                    loss.backward()
                    self.opti.step()

            print(f"\nEpoch: {epoch}/{stop_value}, Loss:{loss}")

            Acc = self.evaluator.Pixel_Accuracy()
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            print('Validation:')
            # print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
            print('Loss: %.3f' % test_loss)

            torch.save(self.net.state_dict(), self.model)

            if epoch % 50 == 0:
                torch.save(self.net.state_dict(), self.model_copy.format(epoch, loss))
                print("model_copy is saved !")
            if epoch > stop_value:
                break
            epoch += 1

    def validation(self, epoch):
            test_loss = 0.0
            while True:
                for inputs, labels in tqdm(self.val_loader, desc=f"Epoch{epoch}/stop_value", ascii=True,
                                           total=len(self.val_loader)):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    out = self.net(inputs)
                    output = self.th(out)
                    loss = self.loss(output, labels)
                    test_loss += loss.item()
                # tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
                    pred = output.data.cpu().numpy()
                    labels = labels.cpu().numpy()
                    pred = np.argmax(pred, axis=1)
                    labels = np.argmax(labels, axis=1)
                # Add batch sample into evaluator
                    self.evaluator.add_batch(labels, pred)

            # Fast test during the training
                Acc = self.evaluator.Pixel_Accuracy()
                Acc_class = self.evaluator.Pixel_Accuracy_Class()
                mIoU = self.evaluator.Mean_Intersection_over_Union()
                FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
                print('Validation:')
                # print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
                print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
                print('Loss: %.3f' % test_loss)

if __name__ == '__main__':
    path = '../data/liver/train/'
    model = U_net(n_channels=3, n_classes=3)
    t = trainer(r"../data/liver", r'./model.plt', r'./model_{}_{}.plt', img_save_path=r'../result_res_1')
    # t = trainer(r"../data/liver/train", r'./model.plt', r'./model_{}_{}.plt', img_save_path=r'./at1')
    t.train(300)
