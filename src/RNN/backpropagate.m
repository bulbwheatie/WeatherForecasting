% Updates the weight matrices by back propagating errors
%
% X = [s x d] d is features, s is nubmer of samples (also stacks)
% 
% signals = [n x s] where n is the nubmer of neurons and s is the number of
% stacks. Each level of neurons has a corresponding column vector of
% signals. [n x 1] is the signal for the first stack's hidden layer
% Y = [m x d] where the mth sample is the final output

function [Uinput, Uprev, Uoutput] = backpropagate(X, Y, signals, Ypred, Winput, Wprev, Woutput, lambda)
    n = size(Wprev, 1); %number of neurons
    l = size(Y, 2); %number of features in the output
    m = size(X, 1); %number of samples
    s = size(signals, 2); %number of stacks
    
    % Calculate the deltas for all the layers    
    % For each stack layer, calculate the output error
    % DjN = [s x L] deltas for the final output vector (linear node)
    % i.e. DjN(1,1) corresponds to the delta term for the 1st output
    % feature of the 1st stack 
    DjN = zeros(s, l);
    for i = 1:s
        DjN(i,:) = Y(i, :) - Ypred(i, :);
    end
    
    % DiT = [1 x n] deltas for the hidden nodes in the final layer
    % (squashing present)
    % DiT(1, 1) correponds to the delta term for the first neuron in the
    % hidden layer prior to the output node
    % NEED - potential to the neuron, weight matrix (row) that applies to its
    % outputs, DjT values (also a row)
    DiN = zeros(s, n);
    for i = 1:n
        DiN(s, i) = sum(Woutput(i, :) .* DjN) * (1 + tanh(signals(:,s))*tanh(signals(:,s)).');
    end
    
    % Calculate delta terms for each hidden neuron in each stack prior to
    % the final output one
    % DiN = [s x n] deltas for each hidden layer
    % DiN(1,2) delta for the first neuron in the 2nd stack layer
    for t = s-1:-1:1
        for i = 1:n
            DiN(t, i) = (sum(DiN(t+1,:) .* Wprev(i,:)) + sum(DjN(t,:) .* Woutput(i,:))) * (1 + tanh(signals(:,t))*tanh(signals(:,t)).');
        end
    end
    
    % Calculate update the weights based on these deltas
    Uinput = zeros(size(Winput));
    for i=1:size(Winput, 1)
        for j=1:size(Winput,2)
            tmpSum = 0;
            for t=1:s
               tmpSum = tmpSum + X(t,i) * DiN(t,j);
            end
            Uinput(i,j) = Winput(i,j) + lambda * tmpSum;
        end
    end
    
    Uprev = zeros(n,n);
    for i=1:size(Winput, 1)
        for j=1:size(Winput,2)
            tmpSum = 0;
            for t=2:s
               tmpSum = tmpSum + signals(i, t-1) * DiN(t,j);
            end
            Uprev(i,j) = Wprev(i,j) + lambda * tmpSum;
        end
    end
    
    Uoutput = zeros(size(Woutput));
    for i=1:size(Winput, 1)
        for j=1:size(Winput,2)
            tmpSum = 0;
            for t=1:s
               tmpSum = tmpSum + signals(i, t) * DiN(t,j);
            end
            Uoutput(i,j) = Woutput(i,j) + lambda * tmpSum;
        end
    end
end
