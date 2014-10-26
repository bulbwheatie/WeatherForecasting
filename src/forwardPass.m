% forwardPass.m
% Function that performs a forward pass through one layer of the network
% Multiplies inputs by weight (including bias term) and then passes value
% through signal function (in this case hyperbolic tangebt)

% Inputs
% X = 1 x d matrices
% W = d x p matrix where p is the number of perceptrons in the layer

% Outputs 
% Y = 1 x p outputs from each perceptron in the layer

function Y = forwardPass(X, W)    

    %Add the bias feature to the input matrix
    B = features(X);

    %Add an additional value to weights (should this be random?)
    W = features(W);

    %Apply the non-linearity squashing function
    Y = tanh(B.' * W);

end