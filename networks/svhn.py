"""
This network are inspired to the ones defined in https://github.com/CuthbertCai/pytorch_DANN
Credits to @CuthbertCai

"""

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .rev_grad import grad_reverse as GRL
from .dial import DomainAdaptationLayer as DAL


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


class SVHN_net(nn.Module):

    def __init__(self):
        super(SVHN_net, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv3_drop = nn.Dropout2d()

        self.fc_type = SVHN_net_classifier
        self.domain_discriminator_type = SVHN_Domain_classifier

        self.out_features = 128*3*3

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.conv3(x))
        x = self.conv3_drop(x)

        feat = x.view(-1, 128 * 3 * 3)
        return feat


class SVHN_net_classifier(nn.Module):
    def __init__(self, feat_in=1152, n_classes=10):
        super(SVHN_net_classifier, self).__init__()
        self.fc1 = nn.Linear(feat_in, 3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.fc3 = nn.Linear(2048, n_classes)
        self.dropout = nn.Dropout()

        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        self.fc3.apply(init_weights)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class SVHN_Domain_classifier(nn.Module):

    def __init__(self, feat_in=1152):
        super(SVHN_Domain_classifier, self).__init__()
        self.fc1 = nn.Linear(feat_in, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout()

        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        self.fc3.apply(init_weights)

    def forward(self, x, lam):
        x = GRL(x, lam)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5)

        self.domain_discriminator_type = Domain_classifier
        self.fc_type = LeNet_classifier
        self.out_features = 48 * 4 * 4
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        feat = x.view(-1, 48 * 4 * 4)
        return feat


class Domain_classifier(nn.Module):

    def __init__(self, feat_in=48 * 4 * 4):
        super(Domain_classifier, self).__init__()
        self.fc1 = nn.Linear(feat_in, 100)
        self.fc2 = nn.Linear(100, 1)

        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)

    def forward(self, x, constant):
        x = GRL(x, constant)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class LeNet_classifier(nn.Module):
    def __init__(self, feat_in=48 * 4 * 4, n_classes=10):
        super(LeNet_classifier, self).__init__()
        self.fc1 = nn.Linear(feat_in, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, n_classes)

        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        self.fc3.apply(init_weights)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


def svhn_net():
    return SVHN_net()


def lenet_net():
    return LeNet()
