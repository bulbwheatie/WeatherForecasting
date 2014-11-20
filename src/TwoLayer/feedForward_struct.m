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
function [YPred, signals1, signals1prev, signals2, signals2prev] = feedForward_struct(X, Winput, Winterior, Wprev1, Wprev2, Woutput) 

    n = size(Wprev1, 1);
    s = size(X, 1);
    
    % Initial state has no previous data as inputs
    Xprev1 = zeros(1, n);
    Xprev2 = zeros(1, n);
    
    % TODO - save the signals into each layer
    signals1 = zeros(s, n); 
    signals1prev = zeros(s, n);
    signals2 = zeros(s, n); 
    signals2prev = zeros(s, n);
    
    YPred = zeros(size(X,1), size(X,2)-1, size(X,3));
    % For each sample in the sequence feed through the network    
    for row = 1:size(X, 3);
        for stack = 1:size(X, 1);
            [Xprev1, Xprev2, signals1(stack,:), signals1prev(stack,:), signals2(stack,:), signals2prev(stack,:)] = feedForwardStack_struct(X(stack,:,row), Xprev1, Xprev2, Winput, Winterior, Wprev1, Wprev2);
            YPred(stack, :, row) = Xprev2 * Woutput;
        end
    end
end

