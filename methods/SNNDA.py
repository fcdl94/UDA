import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss import SNNLoss, MultiChannelSNNLoss
import torch
import numpy as np


class Method(nn.Module):
    def __init__(self, network, init_lr, total_batches, device, num_classes=1000, AD=1., AY=0., Td=0.):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.snnl_inv = SNNLoss(inv=True)
        self.snnl = SNNLoss()

        self.network = network
        self.batch = 0
        self.total_batches = total_batches
        self.device = device
        self.AD = AD
        self.AY = AY
        self.T_d = torch.FloatTensor([Td]).to(device)
        self.T_c = torch.tensor([0.]).to(device)

        self.layers = [0, 1]

        feat_size = self.network.out_features
        self.fc = self.network.fc_type(feat_size, num_classes).to(device)

        self.threshold = 0.9
        self.init_lr = init_lr
        # self.start_batch = int(0.1 * total_batches)

    def forward(self, x):
        x = x.to(self.device)

        feat, _ = self.network.forward(x)  # feature vector only
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
        lam = 2. / (1. + np.exp(-10 * p)) - 1

        # if self.batch % 100 == 0:
        #    print(f"Batch {self.batch}, lam {lam}")

        learning_rate = self.init_lr / ((1 + 10 * p) ** 0.75)
        self.optimizer = optim.SGD([
                {'params': self.network.parameters()},
                {'params': self.fc.parameters()}
            ], lr=learning_rate, momentum=0.9)

        self.optimizer.zero_grad()

        self.batch += 1

        inputs_s, targets_s = source_batch

        inputs_s = inputs_s.to(self.device)
        targets_s = targets_s.to(self.device)  # ground truth class scores
        domain_s = torch.zeros(inputs_s.shape[0]).to(self.device)  # source is index 0

        feat_s, layers_s = self.network.forward(inputs_s)  # feature vector only
        prediction = self.fc(feat_s)  # class scores

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

        feat_t, layers_t = self.network.forward(inputs_t)  # feature vector only

        # prediction = self.fc(feat_t)  # class scores for target (not used)
        prediction = nn.functional.softmax(prediction, dim=1)
        scores, predicted = prediction.max(1)
        predicted = torch.where(scores >= self.threshold, predicted, torch.zeros_like(predicted)-1)

        # sum the CE losses
        loss_cl = loss_bx_src

        feats = torch.cat((feat_s, feat_t), 0)
        domains = torch.cat((domain_s, domain_t), 0)
        targets = torch.cat((targets_s, predicted), 0)  # Use pseudo labeling

        class_snnl_loss = self.snnl(feats, targets, T=self.T_c)
        domain_snnl_loss = self.snnl_inv(feats, domains, T=self.T_d)
        domain_snnl_loss_channels = 0.
        for i in self.layers:
            features_to_compare_s = F.adaptive_avg_pool2d(layers_s[i], 1)
            features_to_compare_t = F.adaptive_avg_pool2d(layers_t[i], 1)

            features = torch.cat((features_to_compare_s, features_to_compare_t), 0)

            faetures_to_compare = F.adaptive_avg_pool2d(features, 1).squeeze(-1)

            #print(faetures_to_compare.shape)
            domain_snnl_loss_channels += self.snnl_inv(faetures_to_compare, domains, self.T_d)

        domain_snnl_loss = domain_snnl_loss_channels / len(self.layers)
        class_snnl_loss = self.snnl(feat_s, targets_s, self.T_c)

        loss = loss_cl + lam * self.AY * class_snnl_loss + lam * self.AD * domain_snnl_loss

        loss.backward()
        self.optimizer.step()

        return loss_cl, train_correct_src, train_total_src, domain_snnl_loss, class_snnl_loss
