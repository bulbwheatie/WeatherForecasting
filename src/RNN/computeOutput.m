% Compute the output from a layer

% X = [n x 1] output from the feedforward stack
% Woutput = [n x L] weight for the output layer (L = # of output features)

function Y = computeOutput(X, Woutput) 
    Y = X * Woutput;
end 