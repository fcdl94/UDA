from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss import SNNLoss
import torch
import numpy as np


class Method(nn.Module):
    def __init__(self, network, init_lr, total_batches, device, num_classes=1000, A=1., dim=1024):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCEWithLogitsLoss()
        self.snnl = SNNLoss()

        self.network = network
        self.batch = 0
        self.total_batches = total_batches
        self.device = device

        self.A = A
        self.T_c = torch.tensor([0.]).to(device)

        feat_size = self.network.out_features  # assume all network classifiers are called fc.
        self.domain_discr = self.network.domain_discriminator_type(feat_size).to(device)
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
        self.domain_discr.eval()
        self.fc.eval()

    def observe(self, source_batch, target_batch):
        self.network.train()
        self.domain_discr.train()
        self.fc.train()

        p = float(self.batch) / self.total_batches
        lam = 2. / (1. + np.exp(-10 * p)) - 1
        learning_rate = self.init_lr / ((1 + 10 * p) ** 0.75)
        self.optimizer = optim.SGD([
                {'params': self.network.parameters()},
                {'params': self.domain_discr.parameters(), 'lr': learning_rate*10},
                {'params': self.fc.parameters(), 'lr': learning_rate*10}
            ], lr=learning_rate, momentum=0.9)

        self.optimizer.zero_grad()

        self.batch += 1

        # train the source!
        inputs_s, targets_s = source_batch

        inputs_s = inputs_s.to(self.device)
        targets_s = targets_s.to(self.device)  # ground truth class scores
        domain_s = torch.zeros(inputs_s.shape[0], 1).to(self.device)  # source is index 0

        feat_s = self.network.forward(inputs_s)  # feature vector only
        domain_pred_s = self.domain_discr(feat_s, lam)
        prediction = self.fc(feat_s)  # class scores

        loss_bx_src = self.criterion(prediction, targets_s)  # CE loss
        loss_dm_src = self.domain_criterion(domain_pred_s, domain_s)  # CE loss

        # collect source statistics
        _, predicted = prediction.max(1)
        tr_tot = targets_s.size(0)  # only on target
        tr_crc = predicted.eq(targets_s).sum().item()  # only on target

        train_total_src = tr_tot
        train_correct_src = tr_crc

        # train the target!
        inputs_t, targets_t = target_batch

        inputs_t, targets_t = inputs_t.to(self.device), targets_t.to(self.device)  # class gt
        domain_t = torch.ones(inputs_t.shape[0], 1).to(self.device)  # target is index 1

        feat_t = self.network.forward(inputs_t)  # feature vector only
        domain_pred_t = self.domain_discr(feat_t, lam)
        # prediction = self.fc(feat_t)  # class scores  -> NOT USED.

        loss_dm_tar = self.domain_criterion(domain_pred_t, domain_t)  # CE loss

        # sum the losses and backprop!
        loss_cl = loss_bx_src
        loss_dm = (loss_dm_src + loss_dm_tar)

        loss = loss_cl + self.A * loss_dm

        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            class_snnl_loss = self.snnl(feat_s, targets_s, self.T_c)

        return loss_cl, train_correct_src, train_total_src, loss_dm, class_snnl_loss

