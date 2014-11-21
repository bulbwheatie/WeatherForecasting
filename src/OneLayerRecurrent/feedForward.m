% Performs a feedforward pass through the network

% X = [m x d] sequence of data samples for t = 1,..., m
% Winput = [d x n] input weight matrix (n = number of neurons per layer)
% Wprev = [n x n] weight matrix for signals between layers
% Woutput = [n x L] weight matrix for output(L: # of output features)

% signals = [s x n] input values to the squashing function for each hidden
% layer in the forward pass
function [Y, signals] = feedForward(X, Winput, Wprev, Woutput) 

    s = size(X, 1);
    n = size(Wprev, 1);

    % Initial state has no previous data as inputs
    Xprev = zeros(1, size(Wprev, 1));
    
    % TODO - save the signals into each layer
    signals = zeros(s, n); 
    
    Y = zeros(s, size(Woutput, 2));
    % For each sample in the sequence feed through the network    
    for i = 1:size(X, 1)
        [Xprev, signals(i,:)] = feedForwardStack(X(i,:), Xprev, Winput, Wprev);
        Y(i,:) = computeOutput(Xprev, Woutput);
    end 
end

