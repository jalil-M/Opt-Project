import pandas as pd
import numpy as np
from sklearn import preprocessing

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

def build_spectrum (data,do_spectrum = False ,spect = 0.5, random_state = 1):
    
    """ Function that build a specific spectrum of the class y = 1 from the bank-additional-full.csv.
    It sets a label encoder for columns that aren't numeric and remove the column that are highly correlated, such as duration.
    It returns one dataframe.
    
        Inputs: - data : Dataframe of raw data
                - spect : Pourcentage of the class y = 0. Default value is 0.5. Must between 0.5 and 1.
                - random_state : Originally set at 1, it is used for sampling. For reproductibility, keep it at the same value.
                - do_spectrum : If not set, the dataset return is the preprocessed data with its original balance. Must be set to True, if we want to build a spectrum.
                
        Output: Dataframe with the desired spectrum."""
    
    assert((spect>=0.5)&(spect<=1)), 'Pourcentage of the spectrum is not in the good range'
    
    onehot_data = data.apply(preprocessing.LabelEncoder().fit_transform)
    onehot_data = onehot_data.drop(columns=['duration'])

    yes_onehot_data = onehot_data.loc[onehot_data['y']== 1]
    no_onehot_data = onehot_data.loc[onehot_data['y']== 0]

    final_size_set = 2*yes_onehot_data.size

    size_set = yes_onehot_data.size + no_onehot_data.size
    spect_no = no_onehot_data.size/size_set
    spect_yes = yes_onehot_data.size/size_set

    if do_spectrum:

        size_no = round((spect)*final_size_set)
        size_yes = final_size_set-size_no
        frac = size_no/no_onehot_data.size
        no_onehot_data = no_onehot_data.sample(frac = frac, random_state=random_state)
        frac = size_yes/yes_onehot_data.size
        yes_onehot_data = yes_onehot_data.sample(frac = frac, random_state=random_state)


        spect_no = no_onehot_data.size/final_size_set
        spect_yes = yes_onehot_data.size/final_size_set

    print("Fraction of No :", spect_no, "Fraction of Yes :", spect_yes, sep='\n')
        
    return pd.concat([no_onehot_data, yes_onehot_data], ignore_index=True)

def plot_ROC (y, pred_y):
    
    """ Function that plot the roc of the finals predictions 
    
    Inputs: - y : vector of the actual data set classification
            - pred_y: vector of predictions of y """
    auc = roc_auc_score(y, pred_y)
    print('ROC AUC=%.3f' % (auc))
    
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y, pred_y)
    
    # plot the roc curve for the model
    pyplot.plot(fpr, tpr, marker='--')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()