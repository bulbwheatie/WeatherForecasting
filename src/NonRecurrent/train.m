% Train function that runs training on the data set and returns a weight
% matrix once it converges

% Inputs 
% data sets (input, output) for train, vaildate and test
% numHidden = scalar; number of hidden layers in the network

% Outputs 
% W = 
% erros = scalar - MSE for the network on the test data
function [Wone, Wtwo, Wfinal, validateError, testError] = train(data, learningRate, numHidden, iterations, Wfinal, Wone, Wtwo)
    %initialize weight matrices with random values if the matrices are not
    %provided
    if nargin < 5
        Wfinal = initWeights(size(data.trainX,2) + 1, size(data.trainY,2), -1/8, 1/8);
        if nargin < 6
            Wone = initWeights(size(data.trainX,2) + 1, size(data.trainX,2), -1/8, 1/8);
            if nargin < 7
                Wtwo = initWeights(size(data.trainX,2) + 1, size(data.trainX,2), -1/8, 1/8);
            end
        end
    end
        
    % Initialize validation error
    validateError = zeros(1,iterations);
    
    % Run the forwardPass to get the signals
    % Adjust weights until we reach convergence
    % TODO: Use validation to check for overtraining/convergence
    iter = 1;
    index = 1;
    while (iter <= iterations) 
        [X1, X2, Y] = forwardPassNetwork(data.trainX(index,:), Wone, Wtwo, Wfinal, numHidden);
        [Wone, Wtwo, Wfinal] = backpropagate(data.trainX(index,:), X1, X2, Y, data.trainY(index,:), Wone, Wtwo, Wfinal, learningRate, numHidden);
        validateError(1, iter) = networkError(data.validateX, data.validateY, Wone, Wtwo, Wfinal, numHidden); 
        strcat('Validate error: ', num2str(validateError(1, iter)))
        index = mod(index,size(data.trainX, 1)) + 1;
        iter = iter + 1;
    end
    
    % Calcualte the error in the network 
    testError = networkError(data.testX, data.testY, Wone, Wtwo, Wfinal, numHidden);
    strcat('Test error: ', num2str(testError))
end