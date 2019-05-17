from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torchvision as tv
from torchvision import transforms
from torchvision.datasets import ImageFolder
import datetime
import time
from data import MNISTM, DoubleDataset, get_index_of_classes
from networks import resnet50, lenet_net, svhn_net
from train import *
from logger import TensorboardXLogger as Log
import os
import argparse
import local_path
from methods import Method

parser = argparse.ArgumentParser()
parser.add_argument('--suffix', default="0", help='The suffix for the name of experiment')
parser.add_argument('-D', default=1, type=float)
parser.add_argument('-Y', default=0, type=float)
parser.add_argument('-T', default=0, type=float)
parser.add_argument('--revgrad', action='store_true')
parser.add_argument('--dataset', default="mnist")
parser.add_argument('--so', action='store_true')
parser.add_argument('-s', '--source', default="p")
parser.add_argument('-t', '--target', default="r")
parser.add_argument('--start_epoch', default=0, type=int)

args = parser.parse_args()

assert not (args.revgrad and args.so), "Please, use only one between Revgrad and SO"

# parameters and utils
device = 'cuda'
ROOT = local_path.ROOT
setting = f"uda-{args.dataset}/{args.source}-{args.target}"

if args.revgrad:
    method_name = 'dann'
elif args.so:
    method_name = "SO"
else:
    method_name = f'snnl-d{args.D:.1f}-t{args.T:.1f}'
method_name += f"_{args.suffix}"

save_name = f"models/{setting}/{method_name}.pth"
os.makedirs(f"models/{setting}/", exist_ok=True)
os.makedirs(f"logs/{setting}/", exist_ok=True)

n_classes = 0


def get_setting():
    global n_classes

    paths = {"p": ROOT + "office/Product",
             "a": ROOT + "office/Art",
             "c": ROOT + "office/Clipart",
             "r": ROOT + "office/Real World"}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Normalize to have range between -1,1 : (x - 0.5) * 2
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])
    # Create data augmentation transform
    augmentation = transforms.Compose([transforms.Resize(256),
                                       transforms.RandomResizedCrop(224, (0.6, 1.)),
                                       transforms.RandomHorizontalFlip(),
                                       transform])

    source = ImageFolder(paths[args.source], augmentation)
    target = ImageFolder(paths[args.target], augmentation, target_transform=transforms.Lambda(lambda y: -1))

    test = ImageFolder(paths[args.target], transform)
    EPOCHS = 60
    n_classes = 65
    net = resnet50(pretrained=True, num_classes=65).to(device)
    batch_size = 32

    # target_loader = DataLoader(target, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=8)
    target_loader = DataLoader(target, batch_size=batch_size, shuffle=True, num_workers=8)
    source_loader = DataLoader(source, batch_size=batch_size, shuffle=True, num_workers=8)

    return target_loader, source_loader, test_loader, net, EPOCHS


if __name__ == '__main__':
    # create the Logger
    log = Log(f'logs/{setting}', method_name)

    # Make the dataset
    target_loader, source_loader, test_loader, net, EPOCHS = get_setting()

    dl_len = max(len(source_loader), len(target_loader))
    total_steps = (EPOCHS) * dl_len

    method = Method(net, total_steps, device, num_classes=n_classes, AD=args.D, AY=args.Y, Td=args.T)

    print("Do a validation before starting to check it is ok...")
    val_loss, val_acc = valid(method, valid_loader=test_loader)
    print(f"Epoch {-1:03d} : Test Loss {val_loss:.6f}, Test Acc {val_acc:.2f}")
    print(f"Result should be random guessing, i.e. {100/n_classes:.2f}% accuracy")

    best_val_loss = val_loss
    best_epoch = -1
    best_val_acc = val_acc
    best_model = torch.save(net.state_dict(),  save_name)

    if args.so:
        loader_lenght = 'source'
    else:
        loader_lenght = 'min'

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
            best_model = torch.save(net.state_dict(),  save_name)

    val_loss, val_acc = valid(method, valid_loader=test_loader, conf_matrix=True, log=log, n_classes=n_classes)
    with open('results.csv', 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')},{setting},{method_name},{EPOCHS},{val_loss},{val_acc},{best_epoch},{best_val_loss},{best_val_acc}\n")

time.sleep(2)
exit()
