import torch.nn as nn
import torch.optim as optim
from loss import SNNLoss
import torch
import numpy as np


class Method(nn.Module):
    def __init__(self, network, init_lr, total_batches, device, num_classes_tar=1000, num_classes_src=1000):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.dom_criterion = nn.BCEWithLogitsLoss()

        self.network = network
        self.batch = 0
        self.total_batches = total_batches
        self.device = device

        feat_size = self.network.out_features  # assume all network classifiers are called fc.
        self.fc_tar = self.network.fc_type(feat_size, num_classes_tar).to(device)
        self.fc_src = self.network.fc_type(feat_size, num_classes_src).to(device)
        self.fc_shared = self.network.fc_type(feat_size, num_classes_tar).to(device)

        self.fc_cum = self.network.fc_type(feat_size, num_classes_src).to(device)

        self.shared_classes = num_classes_tar
        # self.fc_cum = self.network.fc_type(feat_size, num_classes_src).to(device)

        self.domain_discriminator = self.network.domain_discriminator_type(feat_size).to(device)

        self.init_lr = init_lr
        self.optimizer = optim.SGD([
                {'params': self.network.parameters()},
                {'params': self.fc_tar.parameters()},
                {'params': self.fc_src.parameters()},
                {'params': self.fc_shared.parameters()},
        ], lr=self.init_lr, momentum=0.9)

    def forward(self, x):
        x = x.to(self.device)

        feat = self.network.forward(x)  # feature vector only
        prediction = self.fc_cum(feat)  # class scores
        _, predicted = prediction.max(1)
        return predicted, prediction

    def extract(self, x):
        x = x.to(self.device)

        feat = self.network.forward(x)  # feature vector only

        return feat

    def fine_tune(self, source_batch, target_batch):
        self.fc_cum.train()
        optimizer = optim.SGD(self.fc_cum.parameters(), lr=self.init_lr, momentum=0.9)

        optimizer.zero_grad()
        inputs, targets = source_batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        feat = self.network.forward(inputs)  # feature vector only
        prediction = self.fc_cum(feat)  # class scores

        loss_bx_src = self.criterion(prediction, targets)  # CE loss

        _, predicted = prediction.max(1)
        tr_tot = targets.size(0)  # only on target
        tr_crc = predicted.eq(targets).sum().item()  # only on target

        train_total = tr_tot
        train_correct = tr_crc

        inputs, targets = target_batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        feat = self.network.forward(inputs)  # feature vector only
        prediction = self.fc_cum(feat)  # class scores

        loss_bx_tar = self.criterion(prediction, targets)  # CE loss

        _, predicted = prediction.max(1)
        tr_tot = targets.size(0)  # only on target
        tr_crc = predicted.eq(targets).sum().item()  # only on target

        train_total += tr_tot
        train_correct += tr_crc

        loss = loss_bx_src + loss_bx_tar
        loss.backward()
        optimizer.step()

        return loss, train_correct, train_total

    def observe(self, source_batch, target_batch):
        self.network.train()
        self.fc_tar.train()
        self.fc_src.train()
        self.fc_shared.train()

        self.optimizer.zero_grad()
        self.batch += 1

        p = float(self.batch) / self.total_batches
        lam = 2. / (1. + np.exp(-10 * p)) - 1

        # SOURCE #######
        inputs, targets = source_batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        targets_filtered = targets[targets < self.shared_classes]
        
        feat = self.network.forward(inputs)  # feature vector only
        prediction = self.fc_src(feat)  # class scores

        feat_filtered = feat[targets < self.shared_classes] 
        prediction_ = self.fc_shared(feat_filtered)  # class scores

        pred_dom = self.domain_discriminator(feat_filtered, lam)

        loss_bx_src = self.criterion(prediction, targets)  # CE loss
        loss_bs_shared_src = self.criterion(prediction_, targets_filtered) 

        loss_dom_src = self.dom_criterion(pred_dom, torch.zeros_like(pred_dom))

        # TARGET #######
        inputs, targets = target_batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        feat = self.network.forward(inputs)  # feature vector only
        prediction = self.fc_tar(feat)  # class scores
        prediction_ = self.fc_shared(feat)  # class scores
        pred_dom = self.domain_discriminator(feat, lam)

        loss_bx_tar = self.criterion(prediction, targets)  # CE loss
        loss_bs_shared_tar = self.criterion(prediction_, targets)
        loss_dom_tar = self.dom_criterion(pred_dom, torch.ones_like(pred_dom))

        # sum the CE losses
        loss_cl = loss_bx_src + loss_bx_tar

        loss = loss_cl + loss_bs_shared_src + loss_bs_shared_tar + loss_dom_src + loss_dom_tar

        if self.batch % 20 == 0:
            print(f"Target Loss: {loss_bx_tar:.4f} "
                  f"T-Shar Loss: {loss_bs_shared_tar:.4f} \n"
                  f"Source Loss: {loss_bx_src:.4f} "
                  f"S-Shar Loss: {loss_bs_shared_src:.4f} \n"
                  f"T-Domn Loss: {loss_dom_tar:.4f} "
                  f"S-Domn Loss: {loss_dom_src:.4f} \n")

        loss.backward()
        self.optimizer.step()

        loss, train_correct, train_total = self.fine_tune(source_batch, target_batch)

        return loss, train_correct, train_total, 0., 0.