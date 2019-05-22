from __future__ import print_function

import torch
import config as cf
import matplotlib.pyplot as plt

from PIL import Image
import os
import sys
import time
import argparse

from utilities_general import validate_dir
from utilities_general import Logger


from activation_utils import save_class_activation_images,preprocess_image
from grad_cam import GradCam
from cnn_layer_visualization import CNNLayerVisualization

parser = argparse.ArgumentParser(description='Visualizations CAM and features')
parser.add_argument('--dataset', default='g1020', type=str, help='dataset = [g1020/isic]')

args = parser.parse_args()
base_path = validate_dir(cf.viz_base_folder+'/')
experiment_path = validate_dir(base_path+cf.exp_folder+'/')
plots_path = validate_dir(experiment_path + 'plots/')

# Hyper Parameter settings
resize=227

print("-" * 80)
log_file = os.path.join(experiment_path, "stdout")
print("Logging to {}".format(log_file))
sys.stdout = Logger(log_file)
# Data Uplaod
print('\n[Phase 1] : Data Preparation')
lines = []
paths = []
labels = []
num_classes = 2

print("| Preparing "+ args.dataset +" visualization dataset...")

sys.stdout.write("| ")
lines = [line.rstrip('\n') for line in open(cf.viz_test_file)]
paths = [line.split()[0] for line in lines]
[labels.append(int(line.split()[1])) for line in lines]
if args.dataset == 'g1020':
    num_classes = 2
else:
    num_classes = 3
# Model
print('\n[Phase 2] : Checkpoint restore')

print('| Resuming from checkpoint...')
_, file_name = getNetwork(args)
checkpoint = torch.load(cf.viz_checkpoint, map_location = 'cpu')
net = checkpoint['net']

def visActivations():
    cnn_layer = cf.viz_filter
    filter_pos = cf.viz_filter

    for i, path in enumerate(paths):
        for j in range(num_classes):
            for k in cnn_layer:
                grad_cam = GradCam(net, target_layer=k)
                original_image = Image.open(path).convert('RGB')
                original_image= original_image.resize((resize, resize), Image.ANTIALIAS)
                prep_img = preprocess_image(original_image)
                cam = grad_cam.generate_cam(prep_img, j)
                save_class_activation_images(plots_path, original_image, cam, 'testImage_layer_'+str(k)+'_class_'+str(j)+'_actual_'+str(labels[i])+'_'+str(i))

    # Fully connected layer is not needed

    pretrained_model = net.features
    for i in cnn_layer:
        layer_vis = CNNLayerVisualization(pretrained_model, i, filter_pos)
        layer_vis.visualise_layer_without_hooks(plots_path)

elapsed_time = 0
start_time = time.time()
visActivations()
epoch_time = time.time() - start_time
print('|Visualizations generation time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))


