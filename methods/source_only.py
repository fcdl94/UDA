import torch.nn as nn
import torch.optim as optim
from loss import SNNLoss
import torch
import numpy as np


class Method(nn.Module):
    def __init__(self, network, init_lr, total_batches, device, num_classes=1000, ):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.snnl = SNNLoss()
        self.Tc = torch.tensor([0.0]).to(device)

        self.network = network
        self.batch = 0
        self.total_batches = total_batches
        self.device = device

        feat_size = self.network.out_features  # assume all network classifiers are called fc.
        self.fc = self.network.fc_type(feat_size, num_classes).to(device)

        self.init_lr = init_lr

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

        p = float(self.batch) / self.total_batches
        learning_rate = self.init_lr / ((1 + 10 * p) ** 0.75)
        self.optimizer = optim.SGD([
                {'params': self.network.parameters()},
                {'params': self.fc.parameters(), 'lr': learning_rate*10}
            ], lr=learning_rate, momentum=0.9)

        self.optimizer.zero_grad()
        self.batch += 1

        inputs_s, targets_s = source_batch

        inputs_s = inputs_s.to(self.device)
        targets_s = targets_s.to(self.device)  # ground truth class scores

        feat_s = self.network.forward(inputs_s)  # feature vector only
        prediction = self.fc(feat_s)  # class scores

        loss_bx_src = self.criterion(prediction, targets_s)  # CE loss

        _, predicted = prediction.max(1)
        tr_tot = targets_s.size(0)  # only on target
        tr_crc = predicted.eq(targets_s).sum().item()  # only on target

        train_total_src = tr_tot
        train_correct_src = tr_crc

        # sum the CE losses
        loss_cl = loss_bx_src

        loss = loss_cl

        with torch.no_grad():
            class_loss = self.snnl(feat_s, targets_s, self.Tc)

        loss.backward()
        self.optimizer.step()

        return loss_cl, train_correct_src, train_total_src, 0., class_loss
