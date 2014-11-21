Location Independent Weather Forecasting Readme


This file contains instructions for how to use the two-layer recurrent neural network. All  files discussed here are included in the TwoLayerRecurrent folder. Because the one-layer recurrent network and for the nonrecurrent network do not work as well as the two-layer network we have not included instructions for their use, but they can for the most part be run in the same way.


TO TRAIN THE NETWORK:

If you wish to use the backpropagation update method, run checker_BP.  If you wish to use the Monte Carlo update method, run checker_MC.

Run the appropriate file with either 'one' (if you want to train on one city's data) or 'all' (if you want to train on all the cities' data) as the argument. This will train a network for each of the six output features and save the weight matrices and error matrix in a file either named 'BP.mat' or 'MC.mat', depending on which checker file you ran.

Both checker_BP and checker_MC have several hyperparameters defined at the top of the file that can be changed as desired.

Example:
checker_BP('all');


TO VIEW THE RESULTS:

Load the weight matrices.  Run createGraphs, inputting the index in the data you'd like to start at (this can be between 1 and 173), all the weight matrices, and the error matrix.  This will load data from Lebanon Airport for the month of October 2014 and use the weight matrices for each feature to create a graph of predicted vs. actual values for that feature at the given index in the data. It will also create a graph that has the training error for each feature. All graphs will be created in the 'graphs' folder.

To create graphs using the average of the results from the Monte Carlo update method and the Backpropagation update method, run createGraphs_combination.

Example:
load('BP.mat');
createGraphs(50, Winputs, Winteriors, Wprev1s, Wprev2s, Woutputs, train_errors);


DESCRIPTION OF ALL FUNCTIONS:

backpropagate: computes the deltas for the weight matrix updates based on the current weight matrices and the error from the output.
calculateUpdates: takes the deltas from backpropagate and 
