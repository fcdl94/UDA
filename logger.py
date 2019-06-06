import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE


class TensorboardXLogger:
    def __init__(self, path, name):
        self.writer = SummaryWriter(log_dir=path+"/"+name)
        self.name = name
        self.path = path

    def log_training(self, epoch, train_loss, train_acc, valid_loss, valid_acc, domain_loss, class_loss, **kwargs):
        print(f"Epoch {epoch}\n\t"
              f"Train Loss: {train_loss:.6f} "
              f"Train Acc : {train_acc:.2f}\n\t"
              f"Test Loss: {valid_loss:.6f} "
              f"Test Acc : {valid_acc:.2f}\n\t"
              f"Class loss: {class_loss:.3f} "
              f"Domain loss: {domain_loss:.3f} ")

        self.writer.add_scalar(f'loss/train', train_loss, epoch)
        self.writer.add_scalar(f'loss/valid', valid_loss, epoch)
        self.writer.add_scalar(f'snnl/class', class_loss, epoch)
        self.writer.add_scalar(f'snnl/domain', domain_loss, epoch)
        self.writer.add_scalar(f'acc/train', train_acc, epoch)
        self.writer.add_scalar(f'acc/val', valid_acc, epoch)

        for k in kwargs:
            self.writer.add_scalar(k, kwargs[k], epoch)

    @staticmethod
    def conf_matrix_figure(cm):
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(title=f'Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')

        fig.tight_layout()
        return fig

    def confusion_matrix(self, y, y_hat, n_classes):
        conf = np.zeros((n_classes, n_classes))

        for i in range(len(y)):
            conf[y[i], y_hat[i]] += 1

        cm = conf.astype('float') / (conf.sum(axis=1)+0.000001)[:, np.newaxis]

        fig = self.conf_matrix_figure(cm)
        self.writer.add_figure('conf_matrix', fig, close=True)

        avg_acc = np.diag(cm).mean() * 100.
        print(f"Per class accuracy: {avg_acc}")
        return conf

    def print_tnse(self, method, source_loader, test_loader):
        # compute embeddings
        outputs = []
        targets = []

        with torch.no_grad():
            test_len = 0
            for inputs, target in test_loader:
                inputs = inputs.to(method.device)

                _, output = method.extract(inputs)

                outputs.append(output.cpu())
                targets.append(target)
                test_len += target.shape[0]

        with torch.no_grad():
            for inputs, target in source_loader:
                inputs = inputs.to(method.device)

                _, output = method.extract(inputs)

                outputs.append(output.cpu())
                targets.append(target)

        embeddings = torch.cat(outputs)
        labels = torch.cat(targets)

        # make tsne of embeddings (from train and test)

        X = embeddings.numpy()
        y = labels.numpy()
        y[-1] = 0  # trick to use the same colors in the two plots

        print(X.shape)

        tsne = TSNE()
        X_p = tsne.fit_transform(X)

        fig, ax = plt.subplots()
        im = ax.scatter(X_p[test_len:, 0], X_p[test_len:, 1], c=y[test_len:], s=8, alpha=0.8)
        ax.figure.colorbar(im, ax=ax)

        self.writer.add_figure("TSNE/source", fig, close=True)

        fig, ax = plt.subplots()
        im = ax.scatter(X_p[:test_len, 0], X_p[:test_len, 1], c=y[:test_len], s=8, alpha=0.8)
        ax.figure.colorbar(im, ax=ax)

        self.writer.add_figure("TSNE/test", fig, close=True)
