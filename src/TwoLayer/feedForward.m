% Performs a feedforward pass through the network

% X = [m x d+1] sequence of data samples for t = 1,..., m (d is features
% plus a bias)
% Winput = [d+1 x n] input weight matrix (n = number of neurons per layer)
% Winterior = [n+1 x n] weight matrix for signals between layers in a stack
% Wprev1 = [n+1 x n] weight matrix for signals between layer1 between stacks
% Wprev2 = [n+1 x n] weight matrix for signals between layer2 between stacks
% Woutput = [n+1 x L] weight matrix for output(L: # of output features)

% signals1 = [s x n] input values to the squashing function for each hidden
% signals2 = [s x n] input values to the squashing function for each hidden
% signals1prev, signals2prev = signals entering the first and second hidden
% layer respectively from the previous hidden layer
% layer in the forward pass
% Y = outpput from each stack in the network
function [Y, signals1, signals1prev, signals2, signals2prev] = feedForward(X, Winput, Winterior, Wprev1, Wprev2, Woutput) 

    s = size(X, 1);
    n = size(Wprev1, 1);
    L = size(Woutput, 2);
    % Initial state has no previous data as inputs
    Xprev1 = zeros(1, n);
    Xprev2 = zeros(1, n);
    
    % TODO - save the signals into each layer
    signals1 = zeros(s, n); 
    signals1prev = zeros(s, n);
    signals2 = zeros(s, n); 
    signals2prev = zeros(s,n);
    
    Y = zeros(s, L);
    % For each sample in the sequence feed through the network    
    for i = 1:s
        [Xprev1, Xprev2, signals1(i,:), signals1prev(i,:), signals2(i,:), signals2prev(i,:)] = feedForwardStack(X(i,:), Xprev1, Xprev2, Winput, Winterior, Wprev1, Wprev2);
        %Y(i,:) = Xprev2 * Woutput;
        Y(i,:) = (signals2(i,:) + signals2prev(i,:)) * Woutput; %Remove the squashing function from the output layer
    end 
end

