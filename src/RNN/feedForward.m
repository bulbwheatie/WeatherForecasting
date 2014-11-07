% Performs a feedforward pass through the network

% X = [m x d] sequence of data samples for t = 1,..., m
% Winput = [d x n] input weight matrix (n = number of neurons per layer)
% Wprev = [n x n] weight matrix for signals between layers
% Woutput = [n x L] weight matrix for output(L: # of output features)


function Y = feedForward(X, num_output, Winput, Wprev, Woutput) 
   
    % Initial state has no previous data as inputs
    Xprev = zeros(1, size(Wprev, 1));
    
    % TODO - save the signals into each layer
    
    % For each sample in the sequence feed through the network    
    for i = 1:size(X, 1)
        Xprev = feedForwardStack(X(i,:), Xprev, Winput, Wprev);        
    end 
    
    Y = computeOutput(X, Woutput);
end

