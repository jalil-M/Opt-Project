import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from keras.utils.vis_utils import plot_model
from keras_radam import RAdam

from helpers import *

# READ THE DATASET

data = pd.read_csv('bank-additional.csv', sep=';') 

# HELPERS METHODS

def run_benchmark(data, activation = 'selu', N_spec = 30):
    list_optimzers = ['adam', 'sgd', RAdam()]
    spects = np.linspace(0.5,0.99,N_spec)

    sgd_f1 = []
    adam_f1 = []
    radam_f1 = []

    sgd_precision = []
    adam_precision = []
    radam_precision = []

    sgd_recall = []
    adam_recall = []
    radam_recall = []

    for spect in spects:
        one_hot_enc = build_spectrum (data,do_spectrum = True ,spect = spect)
        X = one_hot_enc.drop(columns=['y'])
        y = one_hot_enc['y']
        x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)
        x_train_keras, y_train_keras = build_keras (x_train,y_train)
        model = build_model(activation)
        for l in list_optimzers:
            model.compile(loss='binary_crossentropy', optimizer=l, metrics=[f1_m,precision_m, recall_m])
            model.fit(np.array(x_train_keras), np.array(y_train_keras), epochs=50, batch_size=128, shuffle=True, verbose=0)
            loss, f1_score, precision, recall = model.evaluate(np.array(x_test), np.array(y_test),verbose = 0)

            if l == 'adam':
                adam_f1.append(f1_score)
                adam_precision.append(precision)
                adam_recall.append(recall)

            elif l == 'sgd':
                sgd_f1.append(f1_score)
                sgd_precision.append(precision)
                sgd_recall.append(recall)
            else:
                radam_f1.append(f1_score)
                radam_precision.append(precision)
                radam_recall.append(recall)
    return spects, sgd_f1, sgd_precision, sgd_recall, adam_f1, adam_precision, adam_recall, radam_f1, radam_precision, radam_recall


def run_training_benchmarking(data, epochs = 50):
    list_optimzers = ['adam', 'sgd', RAdam()]
    
    adam_val_loss = []
    radam_val_loss = []
    sgd_val_loss = []
    
    
    one_hot_enc = build_spectrum (data)

    X = one_hot_enc.drop(columns=['y'])
    y = one_hot_enc['y']
    x_keras, y_keras = build_keras (X,y)

    model = build_model(activation='selu')
    for l in list_optimzers:
        model.compile(loss='binary_crossentropy', optimizer=l, metrics=['accuracy'])
        history = model.fit(np.array(x_keras), np.array(y_keras), 
                    epochs=epochs, batch_size=128,
                    validation_split=0.2, shuffle=True, verbose = 0)

        if l == 'adam':
            adam_val_loss.append(history.history['val_loss'])

        elif l == 'sgd':
            sgd_val_loss.append(history.history['val_loss'])
        else:
            radam_val_loss.append(history.history['val_loss'])
    return sgd_val_loss, radam_val_loss,adam_val_loss
                

# RESULTS

spects_tanh, sgd_f1_tanh, sgd_precision_tanh, sgd_recall_tanh, adam_f1_tanh, adam_precision_tanh, adam_recall_tanh, radam_f1_tanh, radam_precision_tanh, radam_recall_tanh = run_benchmark(data,activation='tanh', N_spec=49)

spects_selu, sgd_f1_selu, sgd_precision_selu, sgd_recall_selu, adam_f1_selu, adam_precision_selu, adam_recall_selu, radam_f1_selu, radam_precision_selu, radam_recall_selu = run_benchmark(data, N_spec=49)

# PLOTS

build_plot_benchmark(spects_selu, sgd_f1_selu, adam_f1_selu, radam_f1_selu, 'F1 Score', 'SELU');
build_plot_benchmark(spects_selu, sgd_precision_selu, adam_precision_selu, radam_precision_selu, 'Precision', 'SELU');
build_plot_benchmark(spects_selu, sgd_recall_selu, adam_recall_selu, radam_recall_selu, 'Recall', 'SELU');

build_plot_benchmark(spects_tanh, sgd_f1_tanh, adam_f1_tanh, radam_f1_tanh, 'F1 Score', 'TANH');
build_plot_benchmark(spects_tanh, sgd_precision_tanh, adam_precision_tanh, radam_precision_tanh, 'Precision', 'TANH');
build_plot_benchmark(spects_tanh, sgd_recall_tanh, adam_recall_tanh, radam_recall_tanh, 'Recall', 'TANH');

# TRAINING BENCHMARKS

sgd_val_loss, radam_val_loss, adam_val_loss = run_training_benchmarking(data)

build_validation_loss_plot(adam_val_loss[0], radam_val_loss[0], sgd_val_loss[0]);

