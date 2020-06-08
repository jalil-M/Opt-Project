# EPFL Optimization for Machine Learning 2020
## Inbalanced data sets effect on the learning process
### The impact of data's variance with methods s.a. Radam, Adam and SGD

### Description
In August 2019, a variation of the Adam optimiser is discovered through a new publication named "On the Variance of the Adaptive Learning Rate and Beyond" by Liu, Jiang and He. This optimiser, called RAdam, use an adaptive momentum to quantify the variance that is misleading during the beginning of the learning. As imbalanced data set present severely skewed class distribution, we suspect that RAdam will perform poorly compare to Adam due to this property. The aim is to validate or invalidate this assumption.


### Getting Started
This version was designed for python 3.6.6 or higher. To run the model's calculation, it is only needed to execute the file `run.py`. On the terminal, the command is `python run.py`. 

### Prerequisites

#### Libraries
The following librairies are used:
* [numpy](http://www.numpy.org/) 1.14.3, can be obtained through [anaconda](https://www.anaconda.com/download/)
* [pandas](https://pandas.pydata.org/), also available through anaconda
* [scikit-learn](https://scikit-learn.org/stable/): `pip install -U scikit-learn`
* [keras](https://keras.io/): `pip install Keras`
* [keras_radam](https://pypi.org/project/keras-radam/):`pip install keras-rectified-adam`


#### Code
To launch the code `run.py` use the following code files:
* `helpers.py`: Deal with the creation of the spectrum and the building of the neural network

The `data` folder is also needed to store full data set.

### Additional content

The folder `litterature` contains scientific papers that inspired our project.

### Documentation
* [Class Project](https://github.com/epfml/OptML_course/blob/master/labs/mini-project/miniproject_description.pdf) : Description of the project.
* [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/pdf/1908.03265.pdf) : Paper of the RAdam optimiser.
* [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) : Dataset of the experiement with its features.

### Authors
* Members: Cadillon Alexandre, Hoggett Emma, Moussa Abdeljalil

### Project Status
The project was submitted on the 12 June 2020, as part of the [Optimization for Machine Learning](https://github.com/epfml/OptML_course) course.pypi.org/project/keras-radam/
