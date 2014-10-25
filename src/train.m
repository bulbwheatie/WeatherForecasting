% Train function that runs training on the data set and returns a weight
% matrix once it converges

% Inputs = data sets (input, output) for train, vaildate and test

function [weights, errors] = train(train, validate, test)
    W = initWeights(size(train.output,1), size(train.output,2), -1, 1);
    
    iter = 0;
    learningRate = 0.1;
    while (iter < 500) 
        W = backpropagate(train, W, learningRate);
    end
    errors = networkError(validate, W);
end