% Performs a feedForward through a layer of the stack

% X = [1 x d] input feature vector
% Xprev1 = [1 x n] signals from previous stack layer1
% Xprev2 = [1 x n] signals from previous stack layer2
% Winput = [n x d] weight matrix for the input vector
% Winterior = [n x n] weight matrix for between layer1 and layer2
% Wprev1 = [n x n] weight matrix for between layer1s between stacks
% Wprev2 = [n x n] weight matrix for between layer2s between stacks

% Y1 = [1 x n] squashed signals of layer1
% Y2 = [1 x n] squashed signals of layer2
% signal1 = [1 x n] signals of layer1 prior to squashing
% signal2 = [1 x n] signals of layer2 prior to squashing

function [Y1, Y2, signal1, signal2] = feedForwardStack(X, Xprev1, Xprev2, Winput, Winterior, Wprev1, Wprev2) 
    n = size(Wprev,1);
    signal1 = zeros(1,n);
    signal2 = zeros(1,n);
    for i=1:n
        signal1(1,i) = sum(X * Winput(:, i)) + sum(Xprev1 * Wprev1);
    end
    Y1 = tanh(signal1);
    for i=1:n
        signal1(1,i) = sum(Y1 * Winterior) + sum(Xprev2 * Wprev2);
    end
    Y2 = tanh(signal2);
end