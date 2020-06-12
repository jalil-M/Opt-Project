import os
import sklearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.optimizers import SGD, Adam, Nadam
from keras.utils.vis_utils import plot_model
from keras_radam import RAdam

os.environ['TF_KERAS'] = '1'
os.environ['KERAS_BACKEND'] =  'theano'

from helpers import *
from benchmarking import *

##### READ THE DATASET #####

data = pd.read_csv('datasets/bank-additional-full.csv', sep=';')

##### RESULTS BENCHMARKING #####

spects_selu, sgd_f1_selu, sgd_accuracy_selu, sgd_recall_selu, sgd_precision_selu, adam_f1_selu, adam_accuracy_selu,adam_recall_selu, adam_precision_selu, rms_prop_f1_selu, rms_prop_accuracy_selu, rms_prop_recall_selu, rms_prop_precision_selu = run_benchmark(data)

spects_tanh, sgd_f1_tanh, sgd_accuracy_tanh, sgd_recall_tanh, sgd_precision_tanh, adam_f1_tanh, adam_accuracy_tanh, adam_recall_tanh, adam_precision_tanh, rms_prop_f1_tanh, rms_prop_accuracy_tanh, rms_prop_recall_tanh, rms_prop_precision_tanh = run_benchmark(data, activation='tanh')

##### PLOTS BENCHMARKING #####

#f1-score plots
build_plot_benchmark(spects_selu, sgd_f1_selu, adam_f1_selu, rms_prop_f1_selu, 'F1-Score', 'SELU');
build_plot_benchmark(spects_tanh, sgd_f1_tanh, adam_f1_tanh, rms_prop_f1_tanh, 'F1-Score', 'TANH');

#recall plots
build_plot_benchmark(spects_selu, sgd_recall_selu, adam_recall_selu, rms_prop_recall_selu, 'Recall', 'SELU');
build_plot_benchmark(spects_tanh, sgd_recall_tanh, adam_recall_tanh, rms_prop_recall_tanh, 'Recall', 'TANH');

#precision plots
build_plot_benchmark(spects_selu, sgd_precision_selu, adam_precision_selu, rms_prop_precision_selu, 'Precision', 'SELU');
build_plot_benchmark(spects_tanh, sgd_precision_tanh, adam_precision_tanh, rms_prop_precision_tanh, 'Precision', 'TANH');

#accuracy plots
build_plot_benchmark(spects_selu, sgd_accuracy_selu, adam_accuracy_selu, rms_prop_accuracy_selu, 'Accuracy', 'SELU')
build_plot_benchmark(spects_tanh, sgd_accuracy_tanh, adam_accuracy_tanh, rms_prop_accuracy_tanh, 'Accuracy', 'TANH')

##### TRAINING AND TEST EFFECTS (on 'selu' only) #####

#Loss again epochs
sgd_val_loss, adam_val_loss, rms_val_loss = run_training_benchmarking_loss(data)
build_validation_loss_plot(adam_val_loss, rms_val_loss, sgd_val_loss, '88');

sgd_val_loss, adam_val_loss, rms_val_loss = run_training_benchmarking_loss(data,spect=0.5)
build_validation_loss_plot(adam_val_loss, rms_val_loss, sgd_val_loss, '50');

sgd_val_loss, adam_val_loss, rms_val_loss = run_training_benchmarking_loss(data,spect=0.95)
build_validation_loss_plot(adam_val_loss, rms_val_loss, sgd_val_loss, '95');




