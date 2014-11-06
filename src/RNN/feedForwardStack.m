% Performs a feedForward through a layer of the stack

% X = [m x d] input feature vector
% Xprev = [n x 1] signals from previous stack
% Winput = [d x n] weight matrix for the input vector
% Wprev = [n x n] weight matrix for the previous layer

% Y = [1 x n]

function Y = feedForwardStack(X, Xprev, Winput, Wprev) 
    Y = tanh([X Xprev] * [Winput Wprev]);   

end