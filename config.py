############### Configuration file ###############
import math

start_epoch = 1
num_epochs = 300
batch_size = 128
optim_type = 'Adam'
lr = 0.0001
num_samples = 25
weight_decay = 5e-4
allow_cuda = True
beta_type = "Blundell"
prob_base_folder = 'Probabilistic_Experiments'
det_base_folder = 'Deterministic_Experiments'
hybrid_base_folder = 'Hybrid_Experiments'
exp_folder = 'Experiment_ISIC'
trainFile = './ISIC_Split/Train/dataFile.txt'
testFile = './ISIC_Split/Test/dataFile.txt'

#uncertainty evaulation configuration
models_threshold = 10
percentile_threshold = 40

hybrid_feature_extractor = './G1020_PyTorchAlex/Experiment_MNIST_final/checkpoint/alexnet-_200.t7'

#visualizations configuration
viz_base_folder = 'Visualizations'
viz_bayes_base_folder = 'Bayesian_Visualizations'
viz_test_file = './ISIC_Split/Test/dataFile.txt'
viz_checkpoint = './G1020_PyTorchAlex/Experiment_MNIST_final/checkpoint/alexnet-_200.t7'
viz_filter = 20
viz_layers = [0,1,2,3,4]

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'mnist': (0.1307,),
    'stl10': (0.485, 0.456, 0.406),
    'origa': (0.9202612529395414, 0.5903763364041101, 0.2996141250272969),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'mnist': (0.3081,),
    'stl10': (0.229, 0.224, 0.225),
    'origa': (0.10013841515667797, 0.14710732803253226, 0.14491356297455987),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

def dynamic_lr(init, epoch):
    optim_factor = 1
    if (epoch > 60):
        optim_factor = 500
    elif (epoch > 30):
        optim_factor = 100
    elif (epoch > 10):
        optim_factor = 10
    return init/optim_factor

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
