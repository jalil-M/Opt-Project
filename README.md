# EPFL Optimization for Machine Learning 2020
## Optimizer's impact on the learning process for imbalanced data sets

### Description
Optimizers are key parameters for efficient training on deep neural network. Current adaptive-learning-rate optimizers have significantly improved the optimization time of other widely spread fixed-learning-rate optimizers. For adaptive-learning-rate methods, an undesirably large variance in the early stages of training, due to the limited amount of training samples, might drive the model away from optimal solutions. 
Imbalanced data set, on the other hand, presents a severely skewed class distribution that may lead to a large variation of the gradient during the learning.
This way, we will study the impact of imbalanced data sets on
different optimizers' approach, as we suspect undesired behaviour over imbalanced datasets for well-known optimizer's such as SGD, RMSprop and Adam.


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
* `helpers.py`: Deal with the creation of the spectrum, building of the neural network and the plots
* `benchmarking.py`: Functions used for the benchmarking. It is composed of computation of the loss against the number of epochs and the recall, accuracy, precision and F1-score against the spectrum.

The `datasets` folder is also needed to store the full data set. For the experiment, the data set `bank-additional-full.csv` is used. In this folder, the datasheet `bank-additional-full.csv` is used for the experiment.

### Additional content

The folder `literature` contains scientific papers that inspired our project. The folder `figures` present all the figures plotted for the report. The notebook `DataAnalysis.ipynb` has all the analysis on the raw data distribution.

### Documentation
* [Class Project](https://github.com/epfml/OptML_course/blob/master/labs/mini-project/miniproject_description.pdf) : Description of the project.
* [An overview of gradient descent optimization algorithms](https://arxiv.org/pdf/1609.04747.pdf) : Description of optimizers used in this project.
* [Neural Networks for Machine Learning](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf): RMSprop
* [Adam: a method for stochastic optimization](https://arxiv.org/pdf/1412.6980.pdf): Adam
* [Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent): SGD
* [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) : Dataset of the experiment with its features.

### Authors
* Members: Cadillon Alexandre, Hoggett Emma, Moussa Abdeljalil

### Project Status
The project was submitted on the 12 June 2020, as part of the [Optimization for Machine Learning](https://github.com/epfml/OptML_course).
