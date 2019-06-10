'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import math
from .rev_grad import grad_reverse as GRL
from .block import *
from torch.nn import init
from torchvision import models
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'http://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'http://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'http://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'http://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'http://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class ResNet(nn.Module):

    def __init__(self, block, layers, pretrained=None, num_classes=1000, zero_init_residual=False, branch_dim=256):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.dial = False
        self.bn = nn.InstanceNorm2d

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.bn(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        n_features_in = 512*block.expansion
        self.out_features = n_features_in

        self.fc_type = nn.Linear

        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, DAL):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        if pretrained is not None:
            self.load_state_dict(pretrained, strict=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self.bn(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm=self.bn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm=self.bn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feat = x.view(x.size(0), -1)

        return feat


def resnet18(pretrained=None, num_classes=1000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): Number of classes of the system
    """
    if pretrained is not None:
        pre_model = model_zoo.load_url(model_urls['resnet18'])
        pre_model = {i:pre_model[i] for i in pre_model if 'running' not in i}
    else:
        pre_model = None

    model = ResNet(BasicBlock, [2, 2, 2, 2], pretrained=pre_model, num_classes=num_classes)

    return model


def resnet34(pretrained=None, num_classes=1000):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): Number of classes of the system
    """
    if pretrained is not None:
        pre_model = models.resnet34(True)
    else:
        pre_model = None

    model = ResNet(BasicBlock, [3, 4, 6, 3], pretrained=pre_model, num_classes=num_classes)

    return model


def resnet50(pretrained=None, num_classes=1000):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): Number of classes of the system
    """
    if pretrained is not None:
        pre_model = models.resnet50(True)
    else:
        pre_model = None

    model = ResNet(Bottleneck, [3, 4, 6, 3], pretrained=pre_model, num_classes=num_classes)

    return model


def resnet50_dial(pretrained=None, num_classes=1000):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): Number of classes of the system
    """
    raise NotImplementedError
