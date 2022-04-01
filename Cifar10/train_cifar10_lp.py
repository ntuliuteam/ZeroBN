# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import os
import argparse
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models_lp
from models_lp.vgg import vgg_block

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ZeroBN CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='test', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')
parser.add_argument('--our', type=int, default=0, help='If zerobn method')
parser.add_argument('--interval', type=int, default=2, help='the interval of zero and recovery')
parser.add_argument('--prune_ratio', type=float, default=0.5, help='prune ratio')
parser.add_argument('--zerobn_epoch', type=int, default=40, help='epoches set bn to 0')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(32),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                          ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

if args.refine:
    checkpoint = torch.load(args.refine)
    print(checkpoint['cfg'])
    model= models_lp.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
    model.load_state_dict(checkpoint['state_dict'])

else:
    model = models_lp.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s * torch.sign(m.weight.data))  # L1


def zeroBN():
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)

    index = 0

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # data_num = data_num + 1
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    y, bn_i = torch.sort(bn)

    thre_index = int(total * args.prune_ratio)
    thre = y[thre_index]

    if 'vgg' in args.arch:

        for m in model.modules():
            if isinstance(m, vgg_block):
                for m2 in m.modules():
                    if isinstance(m2, nn.BatchNorm2d):

                        k = m2.weight.data.abs() - thre
                        k = k + k.abs()
                        k = k.sign()
                        m2.weight.data = m2.weight.data * k
                        m2.bias.data = m2.bias.data * k
                        nonzero = torch.nonzero(k).size()[0]
                        all = k.size()[0]

                        if 1.0 * nonzero / all < 0.1:
                            m.shortcut.data[1] = m.shortcut.data[0].abs().sign()
                            m2.weight.data = m2.weight.data * 0
                            m2.bias.data = m2.bias.data * 0

                        else:
                            m.shortcut.data[1] = m.shortcut.data[1] * 0

                        break
    else:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                k = m.weight.data.abs() - thre
                k = k + k.abs()
                k = k.sign()
                m.weight.data = m.weight.data * k
                m.bias.data = m.bias.data * k
                nonzero = torch.nonzero(k).size()[0]
                all = k.size()[0]
                if 1.0 * nonzero / all < 0.15:
                    m.weight.data = m.weight.data * 0
                    m.bias.data = m.bias.data * 0


def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        loss.backward()
        if args.sr:
            updateBN()
        optimizer.step()

        if args.our and ((epoch > args.zerobn_epoch and epoch % args.interval == 0) or epoch == args.epochs - 1):
            zeroBN()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            # data, target = Variable(data, volatile=True), Variable(target)
            data, target = data, target
            output = model(data)
            # test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))
    # if save:


best_prec1 = 0.
best_save = 0.
bn_bck = None
all_acc = []

for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    train(epoch)

    if epoch == args.zerobn_epoch + 1:
        best_prec1 = 0.0

    p = test()
    if (not args.our) or (epoch <= args.zerobn_epoch or epoch % args.interval == 0 or epoch == args.epochs - 1):
        prec1 = p
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filepath=args.save)

print("Best accuracy: " + str(best_prec1))
