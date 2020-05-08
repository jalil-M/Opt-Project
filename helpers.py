import pandas as pd
import numpy as np
from sklearn import preprocessing

def build_spectrum (data, spect, random_state = 1):
    
    """ Function that build a specific spectrum of the class y = 1 from the bank-additional-full.csv.
    It sets a label encoder for columns that aren't numeric and remove the column that are highly correlated, such as duration.
    It returns one dataframe.
        Inputs: - data : Dataframe of raw data
                - spect : Pourcentage of the class y = 1. Must between 0 and 1.
                - random_state : originally set at 1, it is used for sampling. For reproductibility, keep it at the same value.
        Output: Dataframe with the desired spectrum."""
    
    onehot_data = data.apply(preprocessing.LabelEncoder().fit_transform)
    onehot_data = onehot_data.drop(columns=['duration'])
    
    yes_onehot_data = onehot_data.loc[onehot_data['y']== 1]
    no_onehot_data = onehot_data.loc[onehot_data['y']== 0]
    
    size_set = yes_onehot_data.size + no_onehot_data.size
    spect_no = no_onehot_data.size/size_set
    spect_yes = yes_onehot_data.size/size_set
    
    if spect_no > spect:
        
        size_set = yes_onehot_data.size/(1-spect)
        frac = spect*size_set/no_onehot_data.size
        no_onehot_data = no_onehot_data.sample(frac = frac, random_state=random_state)
        
    elif spect_no < spect:
        
        size_set = no_onehot_data.size/spect
        frac = (1-spect)*size_set/yes_onehot_data.size
        yes_onehot_data = yes_onehot_data.sample(frac = frac, random_state=random_state)
        

    spect_no = no_onehot_data.size/size_set
    spect_yes = yes_onehot_data.size/size_set
    
    print("Fraction of No :", spect_no, "Fraction of Yes :", spect_yes, sep='\n')
        
    return pd.concat([no_onehot_data, yes_onehot_data], ignore_index=True)