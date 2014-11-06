% Performs a feedforward pass through the network

% X = [m x d] sequence of data samples for t = 1,..., m
% Winput = [d x n] input weight matrix (n = number of neurons per layer)
% Wprev = [n x n] weight matrix for signals between layers
% Woutput = [n x L] weight matrix for output(L: # of output features)


function Y = feedForward(X, num_output) 
    addpath('..');
    
    X = [ones(size(X,1), 1) X]; % Add bias feature

    Winput = init_weights(size(X, 1), size(X, 2),-1/2, 1/2); % Create a d + 1 x n matrix for the extra bias feature
    Wprev = init_weights(size(X, 2), size(X, 2),-1/2, 1/2);
    Woutput = init_weights(size(X, 2), num_output, -1/2, 1/2);
    
    % Initial state has no previous data as inputs
    Xprev = zeros(1, size(Wprev, 1));
    
    % For each sample in the sequence feed through the network    
    for i = 1:size(X, 1)
        Xprev = feedForwardStack(X(i,:), Xprev, Winput, Wprev);        
    end 
    
    Y = computeOutput(X, Woutput);
end

