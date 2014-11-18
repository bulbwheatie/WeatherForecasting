% Calculates the error for the network
% Takes in the input, weight, target output 

% Runs all the data through the system to determine
% error per sample in the test set

% Input
% X = m x d test data input with m samples and d features
% Y = m x n test data output with m samples and n features
% Wone = d x d weight matrix for the first hidden layer
% Wtwo = d x d weight matrix for the second hidden layer
% Wfinal = d x d weight matrix for the output layer
% numHidden = scalar - number of hidden layers in the network

% Output
% netError = scalar - average mean sqaure error per sample 

function netError = networkError(X, Y, Wone, Wtwo, Wfinal, numHidden) 
    [X1, X2, Ytest] = forwardPassNetwork(X, Wone, Wtwo, Wfinal, numHidden);
    
    % Calculate MSE for all data set
    netError = 0;
    for i = 1:size(Y,1)
        netError = netError + msError(Y(i,:), Ytest(i,:)); 
    end
    netError = netError/size(Y,1);
end