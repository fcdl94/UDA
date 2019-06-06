import torch.nn as nn
import torch.optim as optim
from loss import SNNLoss
import torch
import numpy as np


class Method(nn.Module):
    def __init__(self, network, init_lr, total_batches, device, num_classes=1000):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

        self.network = network
        self.batch = 0
        self.total_batches = total_batches
        self.device = device

        feat_size = self.network.out_features  # assume all network classifiers are called fc.
        self.fc = self.network.fc_type(feat_size, num_classes).to(device)

        self.init_lr = init_lr
        self.optimizer = optim.SGD([
                {'params': self.network.parameters()},
                {'params': self.fc.parameters()}
            ], lr=self.init_lr, momentum=0.9)

    def forward(self, x):
        x = x.to(self.device)

        feat = self.network.forward(x)  # feature vector only
        prediction = self.fc(feat)  # class scores
        _, predicted = prediction.max(1)
        return predicted, prediction

    def eval(self):
        self.network.eval()
        self.fc.eval()

    def observe(self, source_batch, target_batch):
        self.network.train()
        self.fc.train()

        self.optimizer.zero_grad()
        self.batch += 1

        # SOURCE #######
        inputs, targets = source_batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        feat = self.network.forward(inputs)  # feature vector only
        prediction = self.fc(feat)  # class scores

        loss_bx_src = self.criterion(prediction, targets)  # CE loss

        _, predicted = prediction.max(1)
        tr_tot = targets.size(0)  # only on target
        tr_crc = predicted.eq(targets).sum().item()  # only on target

        train_total = tr_tot
        train_correct = tr_crc

        # TARGET #######
        inputs, targets = target_batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        feat = self.network.forward(inputs)  # feature vector only
        prediction = self.fc(feat)  # class scores

        loss_bx_tar = self.criterion(prediction, targets)  # CE loss

        _, predicted = prediction.max(1)
        tr_tot = targets.size(0)  # only on target
        tr_crc = predicted.eq(targets).sum().item()  # only on target

        train_total += tr_tot
        train_correct += tr_crc

        # sum the CE losses
        loss_cl = loss_bx_src + loss_bx_tar

        loss = loss_cl

        loss.backward()
        self.optimizer.step()

        return loss_cl, train_correct, train_total, 0., 0.
