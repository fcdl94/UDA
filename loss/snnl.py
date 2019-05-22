import torch
import torch.nn as nn


def euc_dist(x):
    return torch.norm(x[:, None] - x, dim=2, p=2)


def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


# SNNLoss definition
class SNNLoss(nn.Module):
    def __init__(self, inv=False, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.inv = inv

    def forward(self, x, y, T=None, d=None):  # x 2-D matrix of BxF, y 1-D vector of B
        x = x[y != -1]
        if d is not None:
            d = d[y != -1]
        y = y[y != -1]

        if T is None:
            T = torch.tensor([0.]).to(x.device)

        b = len(y)

        x = x / x.std()
        dist = euc_dist(x)

        # make diagonal mask
        m_den = 1 - torch.eye(b)
        m_den = m_den.float().to(x.device)

        e_dist = (-dist) * torch.pow(10, T)

        den_dist = torch.clone(e_dist)
        den_dist[m_den == 0] = float('-inf')

        # make per class mask
        if self.inv:
            m_num = (y != y.unsqueeze(0).t()).type(torch.int)  # - torch.eye(b, dtype=torch.int).to(y.device)
        else:
            m_num_y = (y == y.unsqueeze(0).t()).type(torch.int) - torch.eye(b, dtype=torch.int).to(y.device)
            if d is not None:
                m_num_d = (d != d.unsqueeze(0).t()).type(torch.int)
                m_num = m_num_d * m_num_y
            else:
                m_num = m_num_y

        # print(m_num)
        num_dist = torch.clone(e_dist)
        num_dist[m_num == 0] = float('-inf')
        # print(num_dist)
        # compute logsumexp
        num = torch.logsumexp(num_dist, dim=1)
        den = torch.logsumexp(den_dist, dim=1)

        if torch.sum(torch.isinf(num)) > 0:
            num = num.clone()
            den = den.clone()
            den[torch.isinf(num)] = 0
            num[torch.isinf(num)] = 0
            # print(torch.bincount(y))

        if torch.sum(torch.isnan(num)) > 0:
            print(x.shape)
            print(x)
            print(num_dist.shape)
            print(num_dist)
            print(den_dist)
            print(num.shape)
            print(num)
            print(den)
            raise Exception()

        return -(num - den).mean()


class MultiChannelSNNLoss(nn.Module):
    def __init__(self, inv=False, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.inv = inv

    def forward(self, x, y, T=None, d=None):  # x 2-D matrix of BxF, y 1-D vector of B
        x = x[y != -1]
        if d is not None:
            d = d[y != -1]
        y = y[y != -1]

        if T is None:
            T = torch.tensor([0.]).to(x.device)

        b = len(y)

        loss_cum = 0
        # x have dimension B, C, 1
        for c in range(x.shape[1]):
            dist = torch.abs(x[:, c] - x[:, c].t())  # now it has form B * B

            # make diagonal mask
            m_den = 1 - torch.eye(b)
            m_den = m_den.float().to(x.device)

            e_dist = (-dist) * torch.pow(10, T)

            den_dist = torch.clone(e_dist)
            den_dist[m_den == 0] = float('-inf')

            # make per class mask

            if self.inv:
                m_num = (y != y.unsqueeze(0).t()).type(torch.int)  # - torch.eye(b, dtype=torch.int).to(y.device)
            else:
                m_num_y = (y == y.unsqueeze(0).t()).type(torch.int) - torch.eye(b, dtype=torch.int).to(y.device)
                if d is not None:
                    m_num_d = (d != d.unsqueeze(0).t()).type(torch.int)
                    m_num = m_num_d * m_num_y
                else:
                    m_num = m_num_y

            num_dist = torch.clone(e_dist)
            num_dist[m_num == 0] = float('-inf')

            # compute logsumexp
            num = torch.logsumexp(num_dist, dim=1)
            den = torch.logsumexp(den_dist, dim=1)

            if torch.sum(torch.isinf(num)) > 0:
                num = num.clone()
                den = den.clone()
                den[torch.isinf(num)] = 0
                num[torch.isinf(num)] = 0
                # print(torch.bincount(y))

            if torch.sum(torch.isnan(num)) > 0:
                print(x.shape)
                print(x)
                print(num_dist.shape)
                print(num_dist)
                print(den_dist)
                print(num.shape)
                print(num)
                print(den)
                raise Exception()

            loss_cum += (num - den).mean()

        return - loss_cum / x.shape[1]