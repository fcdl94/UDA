import torch.nn as nn
import torch.optim as optim
from loss import SNNLoss
import torch
import numpy as np


class Method(nn.Module):
    def __init__(self, network, total_batches, device, AD=1., AY=0., Td=0., branch_dim=256):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.snnl_inv = SNNLoss(inv=True)
        self.snnl = SNNLoss()
        self.optimizer = None # it's updated in observe
        self.network = network
        self.batch = 0
        self.total_batches = total_batches
        self.device = device
        self.AD = AD
        self.AY = AY
        self.T_d = Td
        self.T_c = torch.tensor(0)

        branch_input = self.network.fc.in_features # assume all network classifiers are called fc.
        self.branch = nn.Linear(branch_input, branch_dim)
        # init branch!
        nn.init.xavier_normal_(self.branch.weight)
        nn.init.zeros_(self.branch.bias)

    def forward(self, x):
        x = x.to(self.device)

        feat_s = self.network.forward(x)  # feature vector only
        prediction = self.network.predict(feat_s)  # class scores
        _, predicted = prediction.max(1)
        return predicted, prediction

    def eval(self):
        self.network.eval()
        self.branch.eval()

    def observe(self, source_batch, target_batch):
        self.network.train()
        self.branch.train()

        p = float(self.batch) / self.total_batches
        lam = 2. / (1. + np.exp(-10 * p)) - 1

        learning_rate = 0.001 / ((1 + 10 * p) ** 0.75)
        self.optimizer = optim.SGD(self.network.parameters(), lr=learning_rate, momentum=0.9)
        self.optimizer.zero_grad()

        self.batch += 1

        inputs_s, targets_s = source_batch

        inputs_s = inputs_s.to(self.device)
        targets_s = targets_s.to(self.device)  # ground truth class scores
        domain_s = torch.zeros(inputs_s.shape[0]).to(self.device)  # source is index 0

        feat_s = self.network.forward(inputs_s)  # feature vector only
        branch_s = self.branch(feat_s)
        prediction = self.network.predict(feat_s)  # class scores

        loss_bx_src = self.criterion(prediction, targets_s)  # CE loss

        _, predicted = prediction.max(1)
        tr_tot = targets_s.size(0)  # only on target
        tr_crc = predicted.eq(targets_s).sum().item()  # only on target

        train_total_src = tr_tot
        train_correct_src = tr_crc

        # train the target
        inputs_t, targets_t = target_batch

        inputs_t, targets_t = inputs_t.to(self.device), targets_t.to(self.device)  # class gt
        domain_t = torch.ones(inputs_t.shape[0]).to(self.device)  # target is index 1

        feat_t = self.network.forward(inputs_t)  # feature vector only
        branch_t = self.branch(feat_t)

        prediction = self.network.predict(feat_t)  # class scores for target (not used)

        # sum the CE losses
        loss_cl = loss_bx_src

        branchs = torch.cat((branch_s, branch_t), 0)
        domains = torch.cat((domain_s, domain_t), 0)

        class_snnl_loss = self.snnl(feat_s, targets_s, self.T_c)
        domain_snnl_loss = self.snnl_inv(branchs, domains, self.T_d)

        loss = loss_cl + lam * self.AD * domain_snnl_loss + self.AY * class_snnl_loss

        loss.backward()
        self.optimizer.step()

        return loss_cl, train_correct_src, train_total_src, domain_snnl_loss, class_snnl_loss