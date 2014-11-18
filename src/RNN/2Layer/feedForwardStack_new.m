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

function [Y1, Y2, signal1, signal1prev, signal2, signal2prev] = feedForwardStack_new(X, Xprev1, Xprev2, Winput, Winterior, Wprev1, Wprev2) 
    n = size(Wprev1,1);
    signal1 = ones(1,n); %Don't create signals for the bias neuron
    signal1prev = ones(1,n); 
    signal2 = ones(1,n);
    signal2prev = ones(1,n); 
    for i=1:n-1
        signal1(1,i) = sum(X * Winput(:, i)); 
        signal1prev(1,i) =  sum(Xprev1 * Wprev1(:,i));
    end
    Y1 = tanh(signal1 + signal1prev);
    for i=1:n-1
        signal2(1,i) = sum(Y1 * Winterior(:,i));
        signal2prev(1, i) = sum(Xprev2 * Wprev2(:,i));
    end
    Y2 = tanh(signal2 + signal2prev);
end