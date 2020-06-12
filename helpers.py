import pandas as pd
import numpy as np
from sklearn import preprocessing

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense

from IPython.display import clear_output


def build_spectrum (data,do_spectrum = False ,spect = 0.5, random_state = 1):
    
    """ Function that builds a specific spectrum of the class y = 1 from the full dataset.
    It sets a label encoder for columns that are not numeric and remove the columns that are highly correlated, such as duration.
    It returns one dataframe.
    
        Inputs: - data : Dataframe of raw data
                - spect : Pourcentage of the class y = 0. Default value is 0.5. Must between 0.5 and 1.
                - random_state : Originally set at 1, it is used for sampling. For reproductibility, keep it at the same value.
                - do_spectrum : If not set, the dataset return is the preprocessed data with its original balance. Must be set to True, if we want to build a spectrum.
                
        Output: Dataframe with the desired spectrum."""
    
    assert((spect>=0.5)&(spect<=1)), 'Pourcentage of the spectrum is not in the good range'
    
    encode_data = data.apply(preprocessing.LabelEncoder().fit_transform)
    encode_data = encode_data.drop(columns=['duration'])

    yes_encode_data = encode_data.loc[encode_data['y']== 1]
    no_encode_data = encode_data.loc[encode_data['y']== 0]

    final_size_set = 2*yes_encode_data.size

    size_set = yes_encode_data.size + no_encode_data.size
    spect_no = no_encode_data.size/size_set
    spect_yes = yes_encode_data.size/size_set

    if do_spectrum:

        size_no = round((spect)*final_size_set)
        size_yes = final_size_set-size_no
        frac = size_no/no_encode_data.size
        no_encode_data = no_encode_data.sample(frac = frac, random_state=random_state)
        frac = size_yes/yes_encode_data.size
        yes_encode_data = yes_encode_data.sample(frac = frac, random_state=random_state)


        spect_no = no_encode_data.size/final_size_set
        spect_yes = yes_encode_data.size/final_size_set

    print("Fraction of No :", spect_no, "Fraction of Yes :", spect_yes, sep='\n')
    clear_output(wait=True)
        
    return pd.concat([no_encode_data, yes_encode_data], ignore_index=True)


def build_plot_benchmark(spects, sgd_metric, adam_metric, rms_prop_metric, metric_name, activation):
    
    """ Function that plots the benchmarking plots of the finals predictions based on several metrics. The plots are saved in figures folder.
    
    Inputs: - spects : vector of spectrum steps (provided with a number of steps)
            - sgd_metric : vector of the history of the sgd metric
            - adam_metric : vector of the history of the adam metric
            - rms_metric : vector of the history of the rms-prop metric
            - metric_name : String for the name of the metric plotted
            - activation : String for the model's last layer activation function """
    
    plt.style.use('seaborn-whitegrid')
    plt.plot(spects, sgd_metric, label='SGD')
    plt.plot(spects, adam_metric, label='Adam')
    plt.plot(spects, rms_prop_metric, label='RMSprop')
    plt.xlabel('Balance of the data set[-]')
    plt.ylabel('{}[-]'.format(metric_name))
    plt.title('{} against spectrum balance with {}'.format(metric_name, activation))
    plt.legend(loc='upper right')
    plt.savefig('final/{}-spect-{}.png'.format(metric_name, activation))
    plt.clf()


def build_validation_loss_plot(adam, rms_prop, sgd,spect):
    
    """ Function that plots the validation sets history on a specific metric. The plots are saved in the figures folder. 
    
    Inputs: - adam : vector of the history of the adam metric
            - rms : vector of the history of the rms-prop metric
            - sgd : vector of the history of the sgd metric
            - metric : String for the name of the metric """
    
    plt.style.use('seaborn-whitegrid')
    plt.plot(adam, label='Adam')
    plt.plot(rms_prop, label='RMSprop')
    plt.plot(sgd, label='SGD')
    plt.xlabel('Epochs[-]')
    plt.ylabel('Loss[-]')
    plt.title('Validation loss against epochs for {}% spectrum'.format(spect))
    plt.legend(loc='lower right', frameon = True)
    plt.savefig('final/loss-epochs-{}.png'.format(spect))
    plt.clf()

def build_keras (x,y):
    """ Convert the input data vectors into numpy arrays for the keras model """
    x_keras = np.array(x)
    y_keras = np.array(y)
    y_keras = y_keras.reshape(y_keras.shape[0], 1)
    return x_keras, y_keras

def build_model(activation):
    """Build the neural network. 
    - Input size: 19 neurons
    - 1 layer of 10 neurons and activation function 
    - Output: 1 neuron with sigmoid activation function """
    model = Sequential()
    model.add(Dense(10, input_dim=19, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    return model

def recall_m(y_true, y_pred):
    """Compute the recall metric with the true value and the prediction.
    Input: - y_true: true result from the test set. The value is 0 or 1.
           - y_pred: result of the neural network. The value is 0 or 1.
    Output: Recall of the results, which is true positive on possible positive."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    """Compute the precision metric with the true value and the prediction.
    Input: - y_true: true result from the test set. The value is 0 or 1.
           - y_pred: result of the neural network. The value is 0 or 1.
    Output: Precision of the results, which is true positive on predicted positive."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    """Compute the F1-score metric with the true value and the prediction. 
    It uses the functions precision_m and recall_m.
    Input: - y_true: true result from the test set. The value is 0 or 1.
           - y_pred: result of the neural network. The value is 0 or 1.
    Output: F1-score of the results, which is the recall multiplied by the precision divided by the sum of recall and precision."""
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

    
    
    
