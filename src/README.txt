***********************LOCATION INDEPENDENT WEATHER FORECASTING README***********************


This file contains instructions for how to use the two-layer recurrent neural network. All  files discussed here are included in the TwoLayerRecurrent folder. Because the one-layer recurrent network and for the nonrecurrent network do not work as well as the two-layer network we have not included instructions for their use, but they can for the most part be run in the same way.



TO TRAIN THE NETWORK:

If you wish to use the backpropagation update method, run driver_BP.  If you wish to use the Monte Carlo update method, run driver_MC.

Run the appropriate file with either 'one' (if you want to train on one city's data) or 'all' (if you want to train on all the cities' data) as the argument. This will train a network for each of the six output features and save the weight matrices and error matrix in a file either named 'BP.mat' or 'MC.mat', depending on which driver file you ran.

Both driver_BP and driver_MC have several hyperparameters defined at the top of the file that can be changed as desired.

Example:
>> driver_BP('all');



TO VIEW THE RESULTS:

Load the weight matrices.  Run createGraphs, inputting the index in the data you'd like to start at (this can be between 1 and 173), all the weight matrices, and the error matrix.  This will load data from Lebanon Airport for the month of October 2014 and use the weight matrices for each feature to create a graph of predicted vs. actual values for that feature at the time interval specified by the index. It will also create a graph that has the training error for each feature. All graphs will be created in the 'graphs' folder.

To create graphs using the average of the results from the Monte Carlo update method and the Backpropagation update method, run createGraphs_combination.

Example:
>> load('BP.mat');
>> createGraphs(50, Winputs, Winteriors, Wprev1s, Wprev2s, Woutputs, train_errors);



TO CROSS-VALIDATE ON THE NUMBER OF STACKS PER SAMPLE AND NEURONS PER LAYER:

Run cross_validate_BP, then load and plot the data.

Example:
>> cross_validate_BP
>> load('cross_validate_BP_neurons.mat')
>> plot(neurons, errorn');
>> load('cross_validate_BP_stacks.mat')
>> plot(stacks, errors');



DESCRIPTION OF ALL FUNCTIONS:

backpropagate: computes the deltas for the weight matrix updates based on the current weight matrices and the error from the output.

calculateUpdates: takes the deltas from backpropagate and creates update matrices for each weight matrix.

createGraphs_combination: creates graphs of the predicted vs. actual values of each feature using the average of the output values from the network trained with the Monte Carlo update method and the network trained with backpropagation.

createGraphs: does the same thing as createGraphs_combination, but only uses one network to predict the output.

cross_validate_BP: trains six networks with a fixed number of stacks and various numbers of neurons per layer, then trains six networks with a fixed number of neurons and various numbers of stacks per layer, and saves the final error from each network.  All networks are trained using temperature as the only output.

driver_BP_nonadaptive: trains a network for every feature using backpropagation with a fixed learning rate.

driver_BP: trains a network for every feature using backpropagation with an adaptive learning rate.

driver_MC: trains a network for every feature using the Monte Carlo update method.

feedForward: computes the output for a given sample and weight matrices.

feedForwardStack: performs a feedForward through a single stack.

get_combo_error: finds the mean squared error of the averaged output of a network traind with the Monte Carlo update method and a network trained with backpropagation.

getData_struct: takes either 'all' or 'one' as input to determine whether it will read in the data from all nine cities or just the data from one city. The function reads in and parses the data into a 3d matrix of X values and a 3d matrix of Y values where the stack is on the first dimension, the feature is on the second dimension, and the sample is on the third dimension. It then adds a bias term to the X features. Finally, it randomizes the data samples and uses 50% of them for training data, 25% for validation data, and 25% for test data, and returns a struct containing all the matrices.

getData: The same as getData_struct, but instead of returning a struct of 3d matrices it just returns one 2d matrix of continuous data with the sample on the first dimension and the feature on the second dimension.

initWeights: Creates a random matrix of the given size and with the given range of values.

parseData: reads in raw data from a file, extracts the six desired features from it, zero-means each feature, and divides each feature by its standard deviation.  It returns a matrix with the sample on the first dimension and the feature on the second dimension, and the matrix containing the mean and standard deviation for each feature.

train_BP_nonadaptive: trains a network using backpropagation and a fixed learning rate.

train_BP: trains a network using backpropagation and an adaptive learning rate.

train_MC: trains a network using the Monte Carlo update method.

lookahead/*: contains files for training a network that calculates update values based on error of predictions more than one time interval into the future, rather than using the error from just the next time interval.

