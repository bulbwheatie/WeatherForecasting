% Performs a feedForward through a layer of the stack

% X = [1 x d] input feature vector
% Xprev = [1 x n] signals from previous stack
% Winput = [n x d] weight matrix for the input vector
% Wprev = [n x n] weight matrix for the previous layer

% Y = [1 x n]
% signal = [1 x n] signals prior to squashing

function [Y, signal] = feedForwardStack(X, Xprev, Winput, Wprev) 
    n = size(Wprev,1);
    signal = zeros(1,n);
    for i=1:n
        signal(1,i) = sum(X * Winput(:, i)) + sum(Xprev * Wprev);
    end
    Y = tanh(signal);   
   
end