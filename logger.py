import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


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
        self.writer.add_scalar(f'loss/valid', train_loss, epoch)
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
