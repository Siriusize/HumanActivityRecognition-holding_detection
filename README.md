This git re-enforces this paper with minor structure changes: 

Real-time human activity recognition from accelerometer data using Convolutional Neural Networks 
https://www.sciencedirect.com/science/article/pii/S1568494617305665

Original model is enforced in python2 and tensorflow1.0, which is sort of old-fashion. 
https://github.com/aiff22/HAR

This git uses pytorch and python 3.7 to classify texting while walking from others.

Data is collected from 2 male and 1 female. Training data is consist of 1 male and 1 female while the testing data is consist of the left 1 male. 

If you want to reproduce the whole training procedure, please make sure to install hyper opt. 

conda install -c conda-forge hyperopt

For finding hyper-parameters, 
	python opt.py

For training model,
	python train.py

For predicting,
	python predict.py


