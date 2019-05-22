import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import math
import config as cf
import cv2 as cv

def validate_dir(path):
    if not os.path.isdir(path): os.mkdir(path)
    return path


class Logger(object):
    def __init__(self, output_file):
        self.terminal = sys.stdout
        self.log = open(output_file, "a")

    def flush(self):
        pass

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()


def min_max_normalize(val, min, max):
    return (val - min) / (max - min);



def eval(y_hats, all_labels,all_predictions,all_targets, numClasses, figure_path, prediction_path, eval_path):
    min_val = y_hats.min()
    max_val = y_hats.max()
    certainTH = 6
    y_hats_diff = np.empty((y_hats.shape[0], y_hats.shape[1]))
    # y_hatsSM_diff = np.empty((y_hatsSM.shape[0], y_hatsSM.shape[1]))
    thresholds = []
    for i in range(y_hats.shape[0]):
        for j in range(y_hats.shape[1]):
            classes = y_hats[i][j]
            indexes = classes.argsort()[-2:][::-1]
            first = y_hats[i][j][indexes[0]]
            first = min_max_normalize(first, min_val, max_val)
            second = y_hats[i][j][indexes[1]]
            second = min_max_normalize(second, min_val, max_val)

            y_hats_diff[i][j] = abs(first - second)

    for i in range(y_hats_diff.shape[0]):
        plt.hist(y_hats_diff[i], len(y_hats_diff[i]))
        plt.savefig(figure_path + 'model_'+str(i)+'.png', format='png', dpi=300)
        plt.show()
        plt.close()
        p = np.percentile(y_hats_diff[i], cf.percentile_threshold)
        thresholds.append(p)

    vTH = sum(thresholds)/len(thresholds)

    all_variances = np.empty(y_hats.shape[1])  #variances of the differences of predictions of each sample by given number of models
    certain_correct_stats = np.empty((y_hats.shape[1]))
    certain_wrong_stats = np.empty((y_hats.shape[1]))
    uncertain_stats = np.empty((y_hats.shape[1]))

    for j in range(y_hats.shape[1]): #loop on test items
        t_corr_certain = False
        t_wr_certain = False
        t_uncertain = False
        logits_var = np.empty(y_hats.shape[0])
        corr_certain_models = 0
        wr_certain_models = 0
        uncertain_models = 0
        correct_label = all_labels[j]
        predicted_label = 0
        class_predictions = np.zeros((numClasses, 1))
        for i in range(y_hats_diff.shape[0]): #loop on models
            diff = y_hats_diff[i][j]

            logits_var[i] =  diff
            correct_label = int(all_targets[i][j])

            predicted_label = int(all_predictions[i][j])#np.argmax(y_hats[i][j])
            if diff > vTH:
                class_predictions[predicted_label] += 1

                if correct_label == predicted_label:

                    corr_certain_models += 1
                else:
                    wr_certain_models += 1
            else:
                uncertain_models +=1

        if corr_certain_models + wr_certain_models >= certainTH:
            correct_label =int(all_targets[i][j])

            pred_label = int(all_predictions[i][j])
            if correct_label == pred_label:
                t_corr_certain = True
            else:
                t_wr_certain = True

        else:
            t_uncertain =True

        all_variances[j] =  np.var(logits_var)
        certain_correct_stats[j] = t_corr_certain
        certain_wrong_stats[j] = t_wr_certain
        uncertain_stats[j] = t_uncertain

    string = "Summary\n"
    string += "Threshold for diff variance: " + str(vTH) + "\n"
    string += "Threshold for models: " + str(certainTH) + "\n"
    string += "Total images: " + str(len(all_labels)) + "\n"
    string += "Uncertain for: " + str(np.sum(uncertain_stats)) + "\n"
    string += "Predicted for: " + str(np.sum(certain_correct_stats) + np.sum(certain_wrong_stats)) + "\n"
    string += "Incorrect certain: " + str(np.sum(certain_wrong_stats)) + "\n"
    string += "Correct Certain: " + str(np.sum(certain_correct_stats) ) + "\n"

    string += "Accuracy when predicted: " + str(np.sum(certain_correct_stats)/(np.sum(certain_correct_stats) + np.sum(certain_wrong_stats))) + "\n"


    print(string)
    np.savetxt(os.path.join(prediction_path, 'all_variances_eval2.txt'), all_variances, fmt='%.2f')

    with open(os.path.join(eval_path, 'summary_eval2.txt'), "w") as f:
        f.write(string)

def histogram_eq(img):
    image_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv.equalizeHist(image_yuv[:, :, 0])
    image_rgb = cv.cvtColor(image_yuv, cv.COLOR_YUV2RGB)
    # plt.imshow(image_rgb), plt.axis("off")
    return cv.cvtColor(image_rgb, cv.COLOR_BGR2RGB)