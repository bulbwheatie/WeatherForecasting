% Performs a feedForward through a stack of the network

% X = [1 x d] input feature vector
% Xprev1 = [1 x n] signals from previous stack layer1
% Xprev2 = [1 x n] signals from previous stack layer2
% Winput = [n x d] weight matrix for the input vector
% Winterior = [n+1 x n] weight matrix for between layer1 and layer2
% Wprev1 = [n+1 x n] weight matrix for between layer1s between stacks
% Wprev2 = [n+1 x n] weight matrix for between layer2s between stacks
% +1 for bias

% Y1 = [1 x n] squashed signals of layer1
% Y2 = [1 x n] squashed signals of layer2
% signal1 = [1 x n] signals of layer1 from the same stack prior to squashing
% signal2 = [1 x n] signals of layer2 from the same stack prior to squashing
% signal1prev = [1 x n] signals of layer1 from the previous layer prior to squashing
% signal2prev = [1 x n] signals of layer2 from the previous layer prior to squashing

function [Y1, Y2, signal1, signal1prev, signal2, signal2prev] = feedForwardStack(X, Xprev1, Xprev2, Winput, Winterior, Wprev1, Wprev2) 
    n = size(Wprev1,1);
    signal1 = ones(1,n); %Don't create signals for the bias neuron
    signal1prev = ones(1,n); 
    signal2 = ones(1,n);
    signal2prev = ones(1,n); 
    signal1(1, 1:end-1) = X * Winput;
    signal1prev(1, 1:end-1) = Xprev1 * Wprev1; 
    Y1 = tanh(signal1 + signal1prev);
    signal2(1, 1:end-1) = Y1 * Winterior;
    signal2prev(1, 1:end-1) = Xprev2 * Wprev2;
    Y2 = tanh(signal2 + signal2prev);
end