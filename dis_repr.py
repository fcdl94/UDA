from torch.utils.data import DataLoader, Subset
import torchvision as tv
from torchvision import transforms
from torchvision.datasets import ImageFolder
import datetime
import time
from data import MNISTM
from networks import resnet50, lenet_net, svhn_net
from train import *
from logger import TensorboardXLogger as Log
import os
import argparse
import local_path
from methods import *
from data.common import get_index_of_classes

parser = argparse.ArgumentParser()
parser.add_argument('method_name', help='The name of the experiment')

parser.add_argument('--dataset', default="mnist")
parser.add_argument('-s', '--source', default="p")
parser.add_argument('-t', '--target', default="r")
parser.add_argument('-e', '--epochs', default=None, type=int)

parser.add_argument("-c", "--common_classes", default=0, type=int)

args = parser.parse_args()

# parameters and utils
device = 'cuda' if torch.cuda.is_available() else "cpu"
ROOT = local_path.ROOT
setting = f"disrepr-{args.dataset}/{args.source}-{args.target}"

method_name = args.method_name

save_name = f"models/{setting}/{method_name}.pth"
os.makedirs(f"models/{setting}/", exist_ok=True)
os.makedirs(f"logs/{setting}/", exist_ok=True)

n_classes = 0


def get_setting():
    global n_classes
    global init_lr
    global EPOCHS
    global net

    transform = tv.transforms.Compose([transforms.Resize((28, 28)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    source = tv.datasets.MNIST(ROOT, train=True, download=True,
                               transform=tv.transforms.Compose([
                                   tv.transforms.Grayscale(3),
                                   transform])
                               )
    test = MNISTM(ROOT, train=False, download=True, transform=transform)
    target = MNISTM(ROOT, train=True, download=True, transform=transform)

    indices = get_index_of_classes(torch.tensor(target.targets), list(range(0, 5+args.common_classes)))
    target = Subset(target, indices)

    indices = get_index_of_classes(torch.tensor(source.targets), list(range(5-args.common_classes, 10)))
    source = Subset(source, indices)

    EPOCHS = 40
    net = lenet_net().to(device)
    batch_size = 64
    n_classes = 10
    init_lr = 0.01

    # target_loader = DataLoader(target, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=8)
    target_loader = DataLoader(target, batch_size=batch_size, shuffle=True, num_workers=8)
    source_loader = DataLoader(source, batch_size=batch_size, shuffle=True, num_workers=8)

    return target_loader, source_loader, test_loader


if __name__ == '__main__':
    # create the Logger
    log = Log(f'logs/{setting}', method_name)

    # Make the dataset
    target_loader, source_loader, test_loader = get_setting()

    if args.epochs is not None:
        EPOCHS = args.epochs

    loader_lenght = 'min'
    dl_len = min(len(source_loader), len(target_loader))
    print(f"Num of Batches ({loader_lenght}) is {dl_len}")
    total_steps = EPOCHS * dl_len
    method = NODA(net, init_lr, total_steps, device, num_classes=n_classes)

    print("Do a validation before starting to check it is ok...")
    val_loss, val_acc = valid(method, valid_loader=test_loader)
    print(f"Epoch {-1:03d} : Test Loss {val_loss:.6f}, Test Acc {val_acc:.2f}")
    print(f"Result should be random guessing, i.e. {100/n_classes:.2f}% accuracy")

    best_val_loss = val_loss
    best_epoch = -1
    best_val_acc = val_acc
    # best_model = torch.save(net.state_dict(),  save_name)

    # training loop
    for epoch in range(EPOCHS):

        train_loss, train_acc, dom_loss, class_loss = train_epoch(method, source_loader, target_loader, loader_lenght)

        # valid!
        val_loss, val_acc = valid(method, valid_loader=test_loader)
        print(f"Epoch {epoch + 1:03d} : Test Loss {val_loss:.6f}, Test Acc {val_acc:.2f}")

        log.log_training(epoch, train_loss, train_acc, val_loss, val_acc, dom_loss, class_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            # best_model = torch.save(net.state_dict(),  save_name)

    val_loss, val_acc = valid(method, valid_loader=test_loader, conf_matrix=True, log=log, n_classes=n_classes)
    # with open('results.csv', 'a') as file:
    #    file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')},{setting},{method_name},{EPOCHS},{val_loss},{val_acc},{best_epoch},{best_val_loss},{best_val_acc}\n")

    # log.print_tnse(method, torch.cat([source_loader, target_loader]), "tsne_train")
    log.print_tnse(method, test_loader, "tnse_test")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')},{setting},{method_name},{EPOCHS},{val_loss},{val_acc},{best_epoch},{best_val_loss},{best_val_acc}\n")
    torch.save(net.state_dict(), save_name)

    time.sleep(2)
    exit()
