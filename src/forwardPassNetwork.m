% Calculates the forward pass for the entire network
% Stores the values of signals and outputs from each layer
% Calls forwardPass for each layer in the network

% Inputs:
% X= d x 1 data sample with d features
% Wone = d x p where p is number of perceptrons in layer
% Wtwo = d x p where p is number of perceptrons in layer
% Wfinal = d x p(final) where p is the number of features in final output
% numHidden = number of hidden layers (either scalar = 0, 1 or 2)

% Outputs:
% Y = 1 x p (final)
function [X1, X2, Y] = forwardPassNetwork(X, Wone, Wtwo, Wfinal, numHidden)

    % Zero Hidden Layers
    if (numHidden == 0)
        Y = forwardPass(X, Wfinal);    
    %One Hidden Layer
    elseif (numHidden == 1)
        X1 = forwardPass(X, Wone);
        Y = forwardPass(X1, Wfinal);    
    %Two Hidden Layers
    elseif (numHidden == 2)
        X1 = forwardPass(X, Wone);
        X2 = forwardPass(X1, Wtwo);
        Y = forwardPass(X2, Wfinal);
    end
end