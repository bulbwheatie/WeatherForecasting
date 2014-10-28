% Calculates the backpropagation error and updates the weight matrix for a
% particular layer.

% Calculates the partial derivative for the error, which is a component of
% the signal function and the input value to the signal

% Inputs
% X = d x 1 training sample
% X1 = d x 1 output from hidden first hidden layer
% X2 = d x 1 output from hidden second hidden layer
% Y = d x 1 final output
% Ytarget = d x 1 target output
% Wone = d x p Weights for first hidden layer (p = # of neurons) 
% Wtwo = d x p Weights for second hidden layer (p = # of neurons)
% Wfinal = d x p(final) Weights for final hidden layer (p = features in output)
% numHidden = number of hidden layers (either scalar = 0, 1 or 2)

% Outputs
% Uone = d x p updated weight matrix for first hidden layer
% Utwo = d x p updated weight matrix for second hidden layer
% Ufinal = d x p updated weight matrix for final hidden layer
% error = total amount of error in the network (scalar)

% Returns an updated weight matrix
function [Uone, Utwo, Ufinal] = backpropagate(X, X1, X2, Y, Ytarget, Wone, Wtwo, Wfinal, a, numHidden)
    % Runs the samples through the network 
    % Then recursively calculat the delta values for each perceptron and
    % upate its weight matrix
    Uone = [];
    Utwo = [];
    Ufinal = [];
    
    if (numHidden == 0)
        delta = transpose(features(X)) * finalLayerError(Y, Ytarget);
        Ufinal = Wfinal + delta * a;
    end
    
    % Must provide the initial input X, signals for hidden layer X1
    % Weight matrix for the first hidden layer
    if (numHidden == 1)
        %Calculate for final layer
        deltaPrev = transpose(features(X1)) * finalLayerError(Y, Ytarget);
        Ufinal = Wfinal + deltaPrev * a;    
        
        %Calculate for hidden layer before
        delta = transpose(features(X)) .* hiddenLayerError(deltaPrev, Wfinal, X1);
        Uone = Wone + repmat(delta, [1,size(Wone,2)]) * a;
    end
    
    % Must provide the initial input X, signals for hidden layer X1,
    % signals for the second hidden layer X2
    % Weight matrix for the first and second hidden layer Wone and Wtwo
    if (numHidden == 2)
        %Calculate for final layer
        deltaPrev = transpose(features(X2)) * finalLayerError(Y, Ytarget);
        Ufinal = Wfinal + deltaPrev * a;    
        
        %Calculate for second hidden layer 
        deltaPrev = transpose(features(X1)) .* hiddenLayerError(deltaPrev, Wfinal, X2);
        Utwo = Wtwo + repmat(deltaPrev, [1,size(Wtwo,2)]) * a;
        
                %Calculate for second hidden layer 
        delta = transpose(features(X)) .* hiddenLayerError(deltaPrev, Wtwo, X1);
        Uone = Wone + repmat(delta, [1,size(Wone,2)]) * a;
    end
end