from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import config as cf
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse

from torch.autograd import Variable

import pickle
from utils.Models.DeterministicModels import conv_init
from utils.Models.DeterministicModels import AlexNet_5Conv1FC,AlexNet_5Conv3FC,AlexNet_6Conv3FC
import numpy as np
import pandas as pd
from utils.G1020_DataLoader import G1020_Dataset
from utils.ISIC_DataLoader import ISIC_Dataset
from utilities_general import eval
from utilities_general import validate_dir
from utilities_general import Logger

parser = argparse.ArgumentParser(description='PyTorch Deterministic Model Training')
parser.add_argument('--net_type', default='alexnet_5Conv3FC', type=str, help='model')
parser.add_argument('--dataset', default='g1020', type=str, help='dataset = [mnist/cifar10/cifar100/g1020/isic]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

base_path = validate_dir(cf.det_base_folder+'/')

experiment_path = validate_dir(base_path+cf.exp_folder+'/')
plots_path = validate_dir(experiment_path + 'plots/')
save_ck_path = validate_dir(experiment_path + 'checkpoint/')

eval_path = validate_dir(experiment_path + '/Evaluation')
prediction_path = validate_dir(eval_path + '/Predictions')

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
resize=32
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

trainLoss = []
trainAcc = []
testAcc = []
valAcc = []
testLoss = []
testPredicts = []
testTargets = []
print("-" * 80)
log_file = os.path.join(experiment_path, "stdout")
print("Logging to {}".format(log_file))
sys.stdout = Logger(log_file)

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([])
transform_train = transforms.Compose([])

if (args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    transform_train = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(resize, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(resize, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
    inputs = 3

elif (args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    transform_train = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(resize, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(resize, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100
    inputs = 3


elif (args.dataset == 'mnist'):
    print("| Preparing MNIST dataset...")
    sys.stdout.write("| ")
    transform_train = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(resize, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(resize, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
    inputs = 1



elif (args.dataset == 'g1020'):
    print("| Preparing g1020 dataset...")
    sys.stdout.write("| ")

    num_classes = 2
    inputs = 3
    resize = 227

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = G1020_Dataset(cf.trainFile,resize, transform_train )
    testset = G1020_Dataset(cf.testFile,resize, transform_test )

elif (args.dataset == 'isic'):
    print("| Preparing ISIC dataset...")
    sys.stdout.write("| ")

    num_classes = 3
    inputs = 3
    resize = 227

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = ISIC_Dataset(cf.trainFile,resize, transform_train )
    testset = ISIC_Dataset(cf.testFile,resize, transform_test )

trainloader = torch.utils.data.DataLoader(trainset, batch_size=cf.batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=cf.batch_size, shuffle=False, num_workers=4)

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'alexnet_5Conv1FC'):
        net = AlexNet_5Conv1FC(num_classes,inputs)

    elif (args.net_type == 'alexnet_5Conv3FC'):
        net = AlexNet_5Conv3FC(num_classes,inputs)
    elif (args.net_type == 'alexnet_6Conv3FC'):
        net = AlexNet_6Conv3FC(num_classes,inputs)
    else:
        print('Error : Network not chosen from given choices')
        sys.exit(0)
    file_name = args.net_type

    return net, file_name

# Model
print('\n[Phase 2] : Model setup')

print('| Building net type [' + args.net_type + ']...')
net, file_name = getNetwork(args)
net.apply(conv_init)

if use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
logfile = os.path.join(experiment_path + 'diagnostics_NonBayes{}_{}.txt'.format(args.net_type, args.dataset))

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.Adam(net.parameters(), lr=cf.learning_rate(cf.lr, epoch), weight_decay=cf.weight_decay)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(cf.lr, epoch)))

    for batch_idx, (inputs_value, targets) in enumerate(trainloader):
        if use_cuda:
            inputs_value, targets = inputs_value.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs_value, targets = Variable(inputs_value), Variable(targets)
        outputs = net(inputs_value)               # Forward Propagation
        loss = criterion(outputs, targets)
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.item(), 100.*correct.item()/total))
        sys.stdout.flush()

    trainAcc.append((100*(correct.item()/total)))

    trainLoss.append(loss.item())
    diagnostics_to_write = {'Epoch': epoch, 'Loss': loss.data[0], 'Accuracy': 100*correct / total}
    with open(logfile, 'a') as lf:
        lf.write(str(diagnostics_to_write))

def printCurves(epoch):

    plt.plot(list(range(epoch)) ,np.array(trainLoss))
    plt.savefig( plots_path + 'trainLoss_'+str(epoch)+'_Train.png', format='png', dpi=300)
    plt.show()
    plt.close()

    #Training Accuracy curve
    plt.plot(list(range(epoch)) ,np.array(trainAcc))
    plt.plot(list(range(epoch)), np.array(testAcc))
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(plots_path  + 'trainAccuracy_'+str(epoch)+'_Train.png', format='png', dpi=300)
    plt.show()
    plt.close()


def test(epoch):
    net.eval()
    testTargets = []
    testPredicts = []
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs_value, targets) in enumerate(testloader):
        if use_cuda:
            inputs_value, targets = inputs_value.cuda(), targets.cuda()
        with torch.no_grad():
            inputs_value, targets = Variable(inputs_value), Variable(targets)
        outputs = net(inputs_value)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        testPredicts.extend(predicted.cpu().data.numpy())
        testTargets.extend(targets.cpu().data.numpy())

    testAcc.append((100 * (correct.item() / total)))

    acc = 100.*correct/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))

    test_diagnostics_to_write = {'Validation Epoch': epoch, 'Loss': loss.item(), 'Accuracy': acc}
    with open(logfile, 'a') as lf:
        lf.write(str(test_diagnostics_to_write))

    print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
    state = {
            'net':net if use_cuda else net,
            'acc':acc,
            'epoch':epoch,
    }
    torch.save(state, save_ck_path+file_name+'_'+str(epoch)+'.t7')

print('\n[Phase 3] : Training model')

print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(cf.lr))
print('| Optimizer = ' + str(optim_type))
print('| Batch-size = ' + str(cf.batch_size))


elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)
    printCurves(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Epoch Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

with open(os.path.join(prediction_path, 'trainingLoss.txt'), "wb") as fp:
    pickle.dump(trainLoss, fp)
with open(os.path.join(prediction_path, 'trainingAccuracy.txt'), "wb") as fp:
    pickle.dump(trainAcc, fp)
with open(os.path.join(prediction_path, 'validationAccuracy.txt'), "wb") as fp:
    pickle.dump(testAcc, fp)

print('\n[Done] : Finished')


