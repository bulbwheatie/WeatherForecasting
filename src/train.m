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
    Wone = initWeights(size(data.trainX,2), size(data.trainX,2), -1/2, 1/2);
    Wtwo = initWeights(size(data.trainX,2), size(data.trainX,2), -1/2, 1/2);
    Wfinal = initWeights(size(data.trainX,2), size(data.trainY,2), -1/2, 1/2);
    
    % Run the forwardPass to get the signals
    % Adjust weights until we reach convergence
    % TODO: Use validation to check for overtraining/convergence
    iter = 0;
    while (iter < 10000) 
        [X1, X2, Y] = forwardPassNetwork(data.trainX, Wone, Wtwo, Wfinal, numHidden);
        [Wone, Wtwo, Wfinal] = backpropagate(data.trainX, X1, X2, Y, data.trainY, Wone, Wtwo, Wfinal, learningRate, numHidden);
        validateError = networkError(data.validateX, data.validateY, Wone, Wtwo, Wfinal, numHidden); 
        iter = iter + 1;
        strcat('Validate error: ', num2str(validateError))
    end
    
    % Calcualte the error in the network 
    testError = networkError(data.testX, data.testY, Wone, Wtwo, Wfinal, numHidden);
    strcat('Test error: ', num2str(testError))
end