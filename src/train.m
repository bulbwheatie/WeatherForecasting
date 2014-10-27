% Train function that runs training on the data set and returns a weight
% matrix once it converges

% Inputs 
% data sets (input, output) for train, vaildate and test
% numHidden = scalar; number of hidden layers in the network

% Outputs 
% W = 
% erros = scalar - MSE for the network on the test data
function [Wone, Wtwo, Wfinal, error] = train(data, learningRate, numHidden)

    % Initialzie weight matrices for each hidden layer and the final output
    Wone = initWeights(size(data.trainY,1), size(data.trainY,2), -1, 1);
    Wtwo = initWeights(size(data.trainY,1), size(data.trainY,2), -1, 1);
    Wfinal = initWeights(size(data.trainY,1), size(data.trainY,2), -1, 1);
    
    % Run the forwardPass to get the signals
    % Adjust weights until we reach convergence
    % TODO: Use validation to check for overtraining/convergence
    iter = 0;
    while (iter < 10) 
        [X1, X2, Y] = forwardPassNetwork(data.trainX, Wone, Wtwo, Wfinal, numHidden);
        [Wone, Wtwo, Wfinal, trainError] = backpropagate(data.trainX, X1, X2, Y, data.trainY, Wone, Wtwo, Wfinal, learningRate, numHidden);
        validateError = networkError(data.validateX, data.validateY, Wone, Wtwo, Wfinal, numHidden); 
        iter = iter + 1;
        strcat('Train error: ', num2str(trainError));
        strcat('Validate error: ', num2str(validateError));
    end
    
    % Calcualte the error in the network 
    error = networkError(data.testX, data.testY, Wone, Wtwo, Wfinal, numHidden);
    strcat('Test error: ', num2str(testError));
end