% Performs a feedforward pass through the network

% X = [m x d] sequence of data samples for t = 1,..., m
% Winput = [d x n] input weight matrix (n = number of neurons per layer)
% Winterior = [n x n] weight matrix for signals between layers in a stack
% Wprev1 = [n x n] weight matrix for signals between layer1 between stacks
% Wprev2 = [n x n] weight matrix for signals between layer2 between stacks
% Woutput = [n x L] weight matrix for output(L: # of output features)

% signals1 = [s x n] input values to the squashing function for each hidden
% signals2 = [s x n] input values to the squashing function for each hidden
% layer in the forward pass
function [Y, signals1, signals2] = feedForward_new(X, Winput, Winterior, Wprev1, Wprev2, Woutput) 

    s = size(X, 1);
    n = size(Wprev1, 1);
    L = size(Woutput, 2);
    % Initial state has no previous data as inputs
    Xprev1 = zeros(1, n);
    Xprev2 = zeros(1, n);
    
    % TODO - save the signals into each layer
    signals1 = zeros(s, n); 
    signals2 = zeros(s, n); 
    
    Y = zeros(s, L);
    % For each sample in the sequence feed through the network    
    for i = 1:s
        [Xprev1, Xprev2, signals1(i,:), signals2(i,:)] = feedForwardStack_new(X(i,:), Xprev1, Xprev2, Winput, Winterior, Wprev1, Wprev2);
        Y(i,:) = Xprev2 * Woutput;
    end 
end

