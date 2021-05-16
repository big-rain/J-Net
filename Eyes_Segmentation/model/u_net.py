import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from tensorboardX import SummaryWriter

class up_sampling(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(up_sampling, self).__init__()
        # if bilinear, use the normal convolutions to reduce the numbers of the channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)
        self.conv = Doubleconv(in_channels, out_channels)
    def forward(self, decoder_x1, encoder_x2):
        decoder_x1 = self.up(decoder_x1)
        diffy = torch.tensor([encoder_x2.size()[2] - decoder_x1.size()[2]])
        diffx = torch.tensor([encoder_x2.size()[3] - decoder_x1.size()[3]])

        decoder_x1 = F.pad(decoder_x1, [diffx//2 , diffx-diffx//2, diffy//2, diffy-diffy//2])
        x = torch.cat([encoder_x2, decoder_x1], dim=1)
        return self.conv(x)
class down_sampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down_sampling, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            Doubleconv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)
class Doubleconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Doubleconv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


class U_net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(U_net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = Doubleconv(self.n_channels, 64)
        self.down1 = down_sampling(64, 128)
        self.down2 = down_sampling(128, 256)
        self.down3 = down_sampling(256, 512)
        self.down4 = down_sampling(512, 512)
        self.up1 = up_sampling(1024, 256, bilinear)
        self.up2 = up_sampling(512, 128, bilinear)
        self.up3 = up_sampling(256, 64, bilinear)
        self.up4 = up_sampling(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.th = torch.nn.Sigmoid()
    def forward(self, x):
        # x = self.up(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return self.th(logits)
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.randn(1, 3, 513, 513).to(device)
    # model = U_net(n_channels=3, n_classes=2, bilinear=True).to(device)
    model = U_net(n_channels=3, n_classes=21).to(device)
    output = model(input)
    print(output.shape)
    # summary(model, (3, 256, 256))
    # with SummaryWriter(comment='u_net') as w:
    #     w.add_graph(model, input)

