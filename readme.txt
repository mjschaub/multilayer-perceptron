trainingimages: 5000 training digits, around 500 samples from each digit class. Each digit is of size 28x28, and the digits are concatenated together vertically. The file is in ASCII format, with three possible characters. ' ' means a white (background) pixel, '+' means a gray pixel, and '#' means a black (foreground) pixel. 

traininglabels: a vector of ground truth labels for every digit from trainingimages.

testimages: 1000 test digits (around 100 from each class), encoded in the same format as the training digits.

testlabels: ground truth labels for testimages.

The MLP/NN can be found in multilayer_perceptron.py and the final single perceptron is in perceptron_2.py with an earlier, similarly working version being in perceptron.py
