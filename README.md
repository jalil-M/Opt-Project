# EPFL Optimization for Machine Learning 2020
## Inbalanced data sets effect on the learning process

### Description
In 2016, in the 


### Getting Started
This version was designed for python 3.6.6 or higher. To run the model's calculation, it is only needed to execute the file `run.py`. On the terminal, the command is `python run.py`. 

### Prerequisites

#### Libraries
The following librairies are used:
* [numpy](http://www.numpy.org/) 1.14.3, can be obtained through [anaconda](https://www.anaconda.com/download/)
* [pandas](https://pandas.pydata.org/), also available through anaconda
* [scikit-learn](https://scikit-learn.org/stable/): `pip install -U scikit-learn`
* [keras](https://keras.io/): `pip install Keras`
* [tensor flow](https://www.tensorflow.org/install/): `pip install tensorflow`
* [matplotlib](https://matplotlib.org/3.1.1/users/installing.html): `python -m pip install -U matplotlib`
* [seaborn](https://seaborn.pydata.org/installing.html):`pip install seaborn`


#### Code
To launch the code `run.py` use the following code files:
* `helpers.py`: Deal with the creation of the spectrum and the building of the neural network

The `datasets` folder is also needed to store full data set. In this folder, the data sheet `bank-additional-full.csv` is used for the experiment.

### Additional content

The folder `litterature` contains scientific papers that inspired our project. The folder `figures` present all the figures plotted for the report. The notebook `DataAnalysis.ipynb` has all the analysis on the raw data distribution.

### Documentation
* [Class Project](https://github.com/epfml/OptML_course/blob/master/labs/mini-project/miniproject_description.pdf) : Description of the project.
* [Incorporating Nesterov Momentum Into Adam](https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ) : 
* [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) : Dataset of the experiement with its features.

### Authors
* Members: Cadillon Alexandre, Hoggett Emma, Moussa Abdeljalil

### Project Status
The project was submitted on the 12 June 2020, as part of the [Optimization for Machine Learning](https://github.com/epfml/OptML_course).
