import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from function.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from torchsummary import summary

# from util.ASPP import build_aspp

#####################################################
######################################################
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        low_level_feat = x

        x = self.maxpool(x)
        x = self.layer1(x)
        mid_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        high_level_feat = self.layer4(x)
        return high_level_feat, mid_level_feat, low_level_feat,

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
        # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth')
        # pretrain_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        print('ok')
        self.load_state_dict(state_dict)

def ResNet101(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model

def ResNet50(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model


def ResNet34(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model

def ResNet18(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [2, 2, 2, 2], output_stride, BatchNorm, pretrained=pretrained)
    return model

###############################################################
################################################################

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, in_plannes, out_plannes, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        self.inplanes = in_plannes
        self.out_plannes = out_plannes
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.reduce_dim = nn.Sequential(
            nn.Conv2d(int(self.inplanes / 1), int(self.inplanes / 2), kernel_size = 1, stride=1),
            nn.BatchNorm2d(int(self.inplanes / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(self.inplanes / 2), int(self.inplanes / 4), kernel_size=1, stride=1),
            nn.BatchNorm2d(int(self.inplanes / 4)),
            nn.ReLU(inplace=True),
        )

        self.aspp1 = _ASPPModule(int(self.inplanes / 4), self.out_plannes, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(int(self.inplanes / 4), self.out_plannes, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(int(self.inplanes / 4), self.out_plannes, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(int(self.inplanes / 4), self.out_plannes, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(int(self.inplanes / 4) , self.out_plannes, 1, stride=1, bias=False),
                                             BatchNorm(64),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(320, 64, 1, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x = self.reduce_dim(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = self.bn1(x5)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


#######################################################################
##########################################################################

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
        self.up =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
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
        x = self.up(x)
        logits = self.outc(x)
        # return self.up(logits)
        return logits


#########################################################################
###########################################################################
class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 64
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.conv2 = nn.Conv2d(112, 48, 1, bias=False)
        self.bn2 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(112, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, low_level_feat, mid_level_feat, high_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        mid_level_feat = F.interpolate(mid_level_feat, size=low_level_feat.size()[2:], mode='bilinear',
                                       align_corners=True)
        mid_level_feat = torch.cat((low_level_feat, mid_level_feat), dim=1)

        mid_level_feat = self.conv2(mid_level_feat)
        mid_level_feat = self.bn2(mid_level_feat)
        mid_level_feat = self.relu(mid_level_feat)
        high_level_feat = F.interpolate(high_level_feat, size=mid_level_feat.size()[2:], mode='bilinear',
                                       align_corners=True)
        high_level_feat = torch.cat((high_level_feat, mid_level_feat), dim=1)

        x = self.last_conv(high_level_feat)

        return x
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

########################################################################################################
#######################################################################################################



class L_net(nn.Module):
    def __init__(self, in_channels, out_classes, pretrained=True, out_stride=16 ):
        super(L_net, self).__init__()

        self.in_channels = in_channels
        self.out_classes = out_classes
        self.pretrained = pretrained
        self.out_stride = out_stride

        # backbone set up
        self.backbone = ResNet50(BatchNorm=nn.BatchNorm2d, pretrained=False, output_stride=8)
        # self.backbone = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=False, output_stride=16)

        # multiple features fusion
        self.aspp_hight = ASPP(in_plannes=2048, out_plannes=64, output_stride=16, BatchNorm=nn.BatchNorm2d)
        self.aspp_mid = ASPP(in_plannes=256, out_plannes=64, output_stride=16, BatchNorm=nn.BatchNorm2d)

        # generate mask
        self.generate_u_net = U_net(n_channels=3, n_classes=self.out_classes)

        # decoder
        self.decode = Decoder(backbone='resnet', num_classes=self.out_classes, BatchNorm=nn.BatchNorm2d)
        self.th = torch.nn.Sigmoid()

    def forward(self, input):
        high_level_feat, mid_level_feat, low_level_feat, = self.backbone(input)
        multiple_high_features = self.aspp_hight(high_level_feat)
        multiple_mid_features = self.aspp_mid(mid_level_feat)

        x = self.decode(low_level_feat, multiple_mid_features, multiple_high_features)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        # return multiple_high_features, multiple_mid_features
        x = self.th(x)
        return x






if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=16).to(device)
    model = L_net(in_channels=3, out_classes=3).to(device)
    input = torch.rand(4, 3, 512, 512).to(device)

    # aspp = ASPP(in_plannes=2048, out_plannes=64, output_stride=16, BatchNorm=nn.BatchNorm2d).to(device)
    # input = torch.rand(4, 2048, 32, 32).to(device)


    # mid_level_feat = model(input)
    # print(low_level_feat.shape)
    # print(mid_level_feat.shape)
    # print(high_level_feat.shape)


    summary(model, (3, 512, 512))
    # out = aspp(input)
    # print(out.shape)