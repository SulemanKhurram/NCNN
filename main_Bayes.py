from __future__ import print_function

import os
import sys
import time
import argparse
import datetime
import math
import pickle
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from utilities_general import validate_dir
from utilities_general import Logger
from utilities_general import eval

import config as cf
from utils.G1020_DataLoader import G1020_Dataset
from utils.ISIC_DataLoader import ISIC_Dataset
from utils.BBBlayers import GaussianVariationalInference
from utils.Models.BayesianModels import BBBAlexNet_5Conv1FC,BBBAlexNet_5Conv3FC,BBBAlexNet_6Conv3FC
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch Bayesian Model Training')
parser.add_argument('--net_type', default='alexnet_5Conv3FC', type=str, help='model')
parser.add_argument('--dataset', default='g1020', type=str, help='dataset = [mnist/cifar10/cifar100/g1020/isic]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

base_path = validate_dir(cf.prob_base_folder+'/')

experiment_path = validate_dir(base_path+cf.exp_folder+'/')
plots_path = validate_dir(experiment_path + 'plots/')
save_ck_path = validate_dir(experiment_path + 'checkpoint/')

eval_path = validate_dir(experiment_path + '/Evaluation')
prediction_path = validate_dir(eval_path + '/Predictions')

use_cuda = torch.cuda.is_available()
#torch.cuda.set_device(1)
best_acc = 0
resize=32

trainLoss = []
trainAcc = []
testAcc = []
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
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(32, padding=4),
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
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(32, padding=4),
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
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(32, padding=4),
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
testloader = torch.utils.data.DataLoader(testset, batch_size=cf.batch_size, shuffle=True, num_workers=4)
# Return network & file name

def getNetwork(args):
    if (args.net_type == 'alexnet_5Conv1FC'):
        net = BBBAlexNet_5Conv1FC(num_classes,inputs)

    elif (args.net_type == 'alexnet_5Conv3FC'):
        net = BBBAlexNet_5Conv3FC(num_classes,inputs)
    elif (args.net_type == 'alexnet_6Conv3FC'):
        net = BBBAlexNet_6Conv3FC(num_classes,inputs)
    else:
        print('Error : Network not chosen from given choices')
        sys.exit(0)
    file_name = args.net_type

    return net, file_name


# Model
print('\n[Phase 2] : Model setup')

print('| Building net type [' + args.net_type + ']...')
net, file_name = getNetwork(args)

if use_cuda:
    net.cuda()

vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())
logfile = os.path.join(experiment_path + 'diagnostics_Bayes{}_{}_{}.txt'.format(args.net_type, args.dataset, cf.num_samples))

def printCurves(epoch):

    plt.plot(list(range(epoch)) ,np.array(trainLoss))
    plt.savefig( plots_path + 'trainLoss_'+str(epoch)+'.png', format='png', dpi=300)
    plt.show()
    plt.close()

    #Training Accuracy curve
    plt.plot(list(range(epoch)) ,np.array(trainAcc))
    plt.plot(list(range(epoch)), np.array(testAcc))
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(plots_path + 'trainAccuracy_'+str(epoch)+'.png', format='png', dpi=300)
    plt.show()
    plt.close()


def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    optimizer = optim.Adam(net.parameters(), lr=cf.learning_rate(cf.lr,epoch), weight_decay=cf.weight_decay)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(cf.lr,epoch))),
    m = math.ceil(len(testset) / cf.batch_size)
    for batch_idx, (inputs_value, targets) in enumerate(trainloader):
        x = inputs_value.view(-1, inputs, resize, resize)
        y = targets
        if use_cuda:
            x, y = x.cuda(), y.cuda() # GPU settings

        if cf.beta_type is "Blundell":
            beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
        elif cf.beta_type is "Soenderby":
            beta = min(epoch / (cf.num_epochs // 4), 1)
        elif cf.beta_type is "Standard":
            beta = 1 / m
        else:
            beta = 0
        # Forward Propagation
        x, y = Variable(x), Variable(y)
        outputs, kl = net.probforward(x)
        loss = vi(outputs, y, kl, beta)  # Loss
        optimizer.zero_grad()
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        total += targets.size(0)
        correct += predicted.eq(y.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, cf.num_epochs, batch_idx+1,
                    (len(trainset)//cf.batch_size)+1, loss.item(), ((correct.item()/total))*100))

        sys.stdout.flush()
    trainLoss.append(loss.item())
    trainAcc.append((correct.item()/total) * 100)
    diagnostics_to_write =  {'Epoch': epoch, 'Loss': loss.item(), 'Accuracy': ((correct.item()/total))*100}
    with open(logfile, 'a') as lf:
        lf.write(str(diagnostics_to_write))

def test(epoch):
    global best_acc
    testTargets = []
    testPredicts = []
    net.eval()
    test_loss = 0
    correct = 0
    conf = []
    testlabels = []
    m = math.ceil(len(testset) / cf.batch_size)
    total = 0
    for batch_idx, (inputs_value, targets) in enumerate(testloader):
        x = inputs_value.view(-1, inputs, resize, resize)
        y = targets
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            x, y = Variable(x), Variable(y)
        outputs, kl = net.probforward(x)

        if cf.beta_type is "Blundell":
            beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
        elif cf.beta_type is "Soenderby":
            beta = min(epoch / (cf.num_epochs // 4), 1)
        elif cf.beta_type is "Standard":
            beta = 1 / m
        else:
            beta = 0

        loss = vi(outputs,y,kl,beta)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(y.data).cpu().sum()
        testPredicts.extend(predicted.cpu().data.numpy())
        testTargets.extend(targets.cpu().data.numpy())
        testlabels.extend(targets.cpu().data.numpy())
        predics = F.softmax(outputs, dim=1)
        results = torch.topk(predics.cpu().data, k=1, dim=1)
        conf.append(results[0][0].item())

        predicted.eq(y.data).cpu().sum()

    testAcc.append((100 * (correct.item() / total)))


    print("\nEpoch " + str(epoch)+" Correct predictions:" + str(correct.item()))
    print("\nEpoch " + str(epoch) + " Total:" + str(total))
    acc = 100 * (correct.item() / total)
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))
    test_diagnostics_to_write = {'Validation Epoch':epoch, 'Loss':loss.item(), 'Accuracy': acc}
    with open(logfile, 'a') as lf:
        lf.write(str(test_diagnostics_to_write))

    p_hat=np.array(conf)
    confidence_mean=np.mean(p_hat, axis=0)
    confidence_var=np.var(p_hat, axis=0)
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
    aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)

    print ("Epistemic Uncertainity is: ", epistemic)
    print("Aleatoric Uncertainity is: ", aleatoric)
    print("Mean is: ", confidence_mean)
    print("Variance is: ", confidence_var)

    state = {
            'net':net if use_cuda else net,
            'acc':(100*(correct.item()/total)),
            'epoch':epoch,
    }

    save_point = save_ck_path+args.dataset+os.sep
    if not os.path.isdir(save_point):
        os.mkdir(save_point)
    torch.save(state, save_point+file_name+str(epoch)+'.t7')


def final_evaluation(epoch):
    global best_acc
    testTargets = []
    testPredicts = []
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    y_hats = np.empty((cf.models_threshold, len(testset) ,num_classes ))
    y_hatsSM = np.empty((cf.models_threshold, len(testset) ,num_classes ))
    all_predictions = np.empty((cf.models_threshold, len(testset)))
    all_targets = np.empty((cf.models_threshold, len(testset)))

    testlabels = []
    m = math.ceil(len(testset) / cf.batch_size)
    for i in range(cf.models_threshold):
        outputs_all = []
        outputs_allSM = []
        add_Predictions = []
        add_Targets = []
        for batch_idx, (inputs_value, targets) in enumerate(testloader):
            x = inputs_value.view(-1, inputs, resize, resize)
            y = targets
            if use_cuda:
                x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                x, y = Variable(x), Variable(y)
            outputs, kl = net.probforward(x)

            if cf.beta_type is "Blundell":
                beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
            elif cf.beta_type is "Soenderby":
                beta = min(epoch / (num_epochs // 4), 1)
            elif cf.beta_type is "Standard":
                beta = 1 / m
            else:
                beta = 0

            loss = vi(outputs,y,kl,beta)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(y.data).cpu().sum()
            testPredicts.extend(predicted.cpu().data.numpy())
            add_Predictions.extend(predicted.cpu().data.numpy())
            testTargets.extend(targets.cpu().data.numpy())
            add_Targets.extend(targets.cpu().data.numpy())
            if i ==0:
                testlabels.extend(targets.cpu().data.numpy())
            outputs_all.extend(outputs.cpu().data.numpy())
            soft = torch.nn.Sigmoid()
            outputsSM = soft(outputs)
            outputs_allSM.extend(outputsSM.cpu().data.numpy())

        all_predictions[i] = add_Predictions
        all_targets[i] = add_Targets
        y_hats[i] = outputs_all

        y_hatsSM[i] = outputs_allSM

    with open(os.path.join(prediction_path, 'y_hats.txt'), 'w') as outfile:

        outfile.write('# Array shape: {0}\n'.format(y_hats.shape))
        for data_slice in y_hats:
            np.savetxt(outfile, data_slice, fmt='%-7.2f')
            outfile.write('# New slice\n')
    with open(os.path.join(prediction_path, 'y_hatsSM.txt'), 'w') as outfile:

        outfile.write('# Array shape: {0}\n'.format(y_hatsSM.shape))
        for data_slice in y_hatsSM:
            np.savetxt(outfile, data_slice, fmt='%-7.2f')
            outfile.write('# New slice\n')

    with open(os.path.join(prediction_path, 'test_labels.txt'), "wb") as fp:
        pickle.dump(testlabels, fp)

    test_labels_int = [int(numeric_string) for numeric_string in testlabels]

    eval(y_hats, test_labels_int,all_predictions,all_targets, num_classes, plots_path, prediction_path, eval_path)

    print("Epoch " + str(epoch)+" Correct predictions:" + str(correct.item()))
    print("Epoch " + str(epoch) + " Total:" + str(total))
    acc = 100 * (correct.item() / total)
    print("\n| Evaluation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))
    test_diagnostics_to_write = {'Evaluation Epoch':epoch, 'Loss':loss.item(), 'Accuracy': acc}
    with open(logfile, 'a') as lf:
        lf.write(str(test_diagnostics_to_write))



print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(cf.num_epochs))
print('| Initial Learning Rate = ' + str(cf.lr))
print('| Optimizer = ' + str(cf.optim_type))
print('| Batch-size = ' + str(cf.batch_size))
elapsed_time = 0

for epoch in range(cf.start_epoch, cf.start_epoch+cf.num_epochs):
    start_time = time.time()
    train(epoch)
    test(epoch)
    printCurves(epoch)
    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('|Epoch Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

with open(os.path.join(prediction_path, 'trainingLoss.txt'), "wb") as fp:
    pickle.dump(trainLoss, fp)
with open(os.path.join(prediction_path, 'trainingAccuracy.txt'), "wb") as fp:
    pickle.dump(trainAcc, fp)
with open(os.path.join(prediction_path, 'validationAccuracy.txt'), "wb") as fp:
    pickle.dump(testAcc, fp)

print('\n[Phase 4] : Evaluating model')
start_time = time.time()
final_evaluation(cf.num_epochs)
epoch_time = time.time() - start_time
print('|Final Evaluation Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))


