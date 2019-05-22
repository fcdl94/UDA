import torch
import torch.nn as nn
import itertools


def train_epoch(method, source_loader, target_loader, lenght='min'):
    train_loss = 0.
    train_corr = 0.
    train_tot = 0.
    train_dom_loss = 0.
    train_class_loss = 0.

    if lenght == 'source':
        if len(source_loader) > len(target_loader):
            iterator = zip(source_loader, itertools.cycle(target_loader))
        else:
            iterator = zip(source_loader, target_loader)
    elif lenght == 'target':
        iterator = zip(itertools.cycle(source_loader), target_loader)
    elif lenght == 'max':
        if len(source_loader) > len(target_loader):
            iterator = zip(source_loader, itertools.cycle(target_loader))
        else:
            iterator = zip(itertools.cycle(source_loader), target_loader)
    else:
        iterator = zip(source_loader, target_loader)

    batch_idx = 0
    for source, target in iterator:
        tl, tc, tt, tdl, tcl = method.observe(source, target)
        train_loss += tl
        train_corr += tc
        train_tot += tt
        train_dom_loss += tdl
        train_class_loss += tcl
        batch_idx += 1

    return train_loss/batch_idx, 100.0*train_corr/train_tot, train_dom_loss/batch_idx, train_class_loss/batch_idx


def valid(method, valid_loader, conf_matrix=False, log=None, n_classes=None):
    criterion = nn.CrossEntropyLoss()
    # make validation
    method.eval()

    test_loss = 0
    test_correct = 0
    test_total = 0

    targets_cum = []
    predict_cum = []

    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.to(method.device)
            targets = targets.to(method.device)

            predicted, prediction = method.forward(inputs)

            loss_bx = criterion(prediction, targets)

            test_loss += loss_bx.item()

            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

            targets_cum.append(targets)
            predict_cum.append(predicted)

    # normalize and print stats
    test_acc = 100. * test_correct / test_total
    test_loss /= len(valid_loader)

    if conf_matrix:
        log.confusion_matrix(torch.cat(targets_cum), torch.cat(predict_cum), n_classes)

    return test_loss, test_acc