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

from keras.optimizers import SGD, Adam, RMSprop
from keras.utils.vis_utils import plot_model

from helpers import *

##### BENCHMARKING METHODS #####

def run_benchmark(data, activation = 'selu', N_spec = 25):
    
    """ Function that runs the complete benchmarking method on all our optimizers and metrics (accuracy, precision, recall and f1-score) 
    
    Inputs: - data : complete dataset as a dataframe
            - activation : name of the activation function
            - N_spec : number of spectrum steps
            
    Outputs : - spects : vector of spectrum values
              - sgd, adam and rms-prop history metrics vectors """
    
    list_optimzers = ['rms_prop','adam', 'sgd']
    spects = np.linspace(0.5,0.99,N_spec)
    
    sgd_f1 = []
    adam_f1 = []
    rms_prop_f1 = []
    
    sgd_accuracy = []
    adam_accuracy = []
    rms_prop_accuracy = []
    
    sgd_recall= []
    adam_recall = []
    rms_prop_recall= []
    
    sgd_precision= []
    adam_precision = []
    rms_prop_precision = []
    
    for spect in spects:
        encoded = build_spectrum (data,do_spectrum = True ,spect = spect)
        X = encoded.drop(columns=['y'])
        y = encoded['y']
        X, y= build_keras (X,y)
        for l in list_optimzers:
            if l=='adam':
                print('[INFO] Using Adam')
                opt = Adam()

            elif l == 'rms_prop':
                print('[INFO] Using RMS-prop')
                opt= RMSprop()
            else:
                print('[INFO] Using SGD')
                opt = SGD()
                
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
                
            
            cvscores_f1 = []
            cvscores_accuracy= []
            cvscores_precision = []
            cvscores_recall= []
            for train, test in kfold.split(X, y):
                model = build_model(activation)
                model.compile(loss='binary_crossentropy', optimizer=opt, metrics=
                              [f1_m,'accuracy',recall_m,precision_m])
                model.fit(X[train], y[train], epochs=20, batch_size=128, verbose=0)
                # evaluate the model
                scores = model.evaluate(X[test], y[test], verbose=0)
                cvscores_f1.append(scores[1])
                cvscores_accuracy.append(scores[2])
                cvscores_recall.append(scores[3])
                cvscores_precision.append(scores[4])
            f1_score = np.mean(cvscores_f1)
            accuracy = np.mean(cvscores_accuracy)
            recall = np.mean(cvscores_recall)
            precision = np.mean(cvscores_precision)
            print ("F1-Score:", f1_score, "Accuracy:", accuracy, sep='\n')
            
            if l == 'adam':
                adam_f1.append(f1_score)
                adam_accuracy.append(accuracy)
                adam_recall.append(recall)
                adam_precision.append(precision)
                
            elif l == 'sgd':
                sgd_f1.append(f1_score)
                sgd_accuracy.append(accuracy)
                sgd_recall.append(recall)
                sgd_precision.append(precision)
            else:
                rms_prop_f1.append(f1_score)
                rms_prop_accuracy.append(accuracy)
                rms_prop_recall.append(recall)
                rms_prop_precision.append(precision)

                
    return spects, sgd_f1, sgd_accuracy, sgd_recall, sgd_precision, adam_f1, adam_accuracy,adam_recall, adam_precision, rms_prop_f1, rms_prop_accuracy, rms_prop_recall, rms_prop_precision

################################

def run_training_benchmarking_f1(data, epochs = 50):
    
    """ Function that runs the benchmarking method with the validation history after training on each optimizer regarding the f1-score. 
    
    Inputs: - data : complete dataset as a dataframe
            - epochs : number of epochs
            
    Outputs : - validation history vectors on each optimizer """
    
    list_optimzers = ['rms_prop','adam', 'sgd']
    
    adam_val_f1 = []
    sgd_val_f1 = []
    rms_val_f1 = []
    
    encoded = build_spectrum (data)
    X = encoded.drop(columns=['y'])
    y = encoded['y']
    X, y= build_keras (X,y)

    for l in list_optimzers:
        if l=='adam':
            print('[INFO] Using Adam')
            opt = Adam()
            
        elif l == 'rms_prop':
            print('[INFO] Using RMS-prop')
            opt= RMSprop()
        else:
            print('[INFO] Using SGD')
            opt = SGD()
            
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
                
        for train, test in kfold.split(X, y):
            model = build_model('selu')
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[f1_m])
            history = model.fit(X[train], y[train], validation_data=(X[test],y[test]) ,epochs=epochs, batch_size=128, verbose=0)

            if l == 'adam':
                adam_val_f1.append(history.history['val_f1_m'])

            elif l == 'sgd':
                sgd_val_f1.append(history.history['val_f1_m'])
            else:
                rms_val_f1.append(history.history['val_f1_m'])
            
    return sgd_val_f1,adam_val_f1, rms_val_f1

################################

def run_training_benchmarking_loss(data, spect = 0.88,epochs = 50):
    
    """ Function that runs the benchmarking method with the validation history after training on each optimizer regarding the accuracy. 
    
    Inputs: - data : complete dataset as a dataframe
            - epochs : number of epochs
            - spect : specific spectrum value
            
    Outputs : - validation history vectors on each optimizer """
    
    list_optimzers = ['rms_prop','adam', 'sgd']
    
    adam_val_loss = []
    sgd_val_loss = []
    rms_val_loss = []
    
    
    X = build_spectrum (data,do_spectrum=True, spect=spect)
    train, test= train_test_split(X, test_size=0.2, stratify = X['y'], random_state=1)
    x_train, y_train = build_keras (train.drop(columns=['y']),train['y'])
    x_test, y_test = build_keras (test.drop(columns=['y']),test['y'])

    for l in list_optimzers:
        if l=='adam':
            print('[INFO] Using Adam')
            opt = Adam()
            
        elif l == 'rms_prop':
            print('[INFO] Using RMS-prop')
            opt= RMSprop()
        else:
            print('[INFO] Using SGD')
            opt = SGD()
            
        model = build_model('selu')
        model.compile(loss='binary_crossentropy', optimizer=opt)
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=128,
                validation_data=(x_test, y_test), verbose = 0)

        if l == 'adam':
            adam_val_loss = history.history['val_loss']
        elif l == 'sgd':
            sgd_val_loss = history.history['val_loss']
        else:
            rms_val_loss = history.history['val_loss']
            
    return sgd_val_loss, adam_val_loss, rms_val_loss 