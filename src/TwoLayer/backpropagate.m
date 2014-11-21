% Updates the weight matrices by back propagating errors
% Only looks at the output of the final layer for backpropagation 

% X = [s x d] d is features, s is nubmer of samples (also stacks)
% Winput = [d+1 x N] where d+1 is the number of input features + a bias 
% Winterior = Wprev1 = Wprev2 = [N+1 x N] (same size but different values)
% Woutput = [N+1 x l] where l is the nubmer of output features
% signals1 = [s x n] where n is the nubmer of neurons and s is the number of
% stacks. Each level of neurons has a corresponding column vector of
% signals. [n x 1] is the signal for the first stack's first hidden layer
% The signal is the value prior to squashing; 
% signals2 = [s x n] is the signals for the second hidden layer
% Y = [m x d] where the mth sample is the final output
% lookahead = number of stacks that were predicted and to include in
% backprops

% Outputs
% DjN = [s x L] delta for output node L= output features
% DiN = [s x n] delta for second hidden layer n = number of neurons
% DpN = [s x n] delta for first hidden layer s = number of stacks

function [DjN, DiN, DpN] = backpropagate(X, Y, signals1, signals2, Ypred, Winterior, Wprev1, Wprev2, Woutput)
    n = size(Wprev1, 1); %number of neurons
    l = size(Y, 2); %number of features in the output
    m = size(X, 1); %number of samples
    s = size(signals1, 1); %number of stacks
    
    signals1(:,size(signals1,2)) = 0;
    signals2(:,size(signals2,2)) = 0; %Bis terms have no input signal
    
    % Calculate the deltas for all the layers    
    % For each stack layer, calculate the output error
    % DjN = [s x L] deltas for the final output vector (linear node)
    % i.e. DjN(1,1) corresponds to the delta term for the 1st output
    % feature of the 1st stack 
    DjN = zeros(size(Y));
    DjN(s) = Y(s) - Ypred(s);
    %DjN = Y-Ypred;
    
    % DiT = [1 x n] deltas for the hidden nodes in the final layer
    % (squashing present)
    % DiT(1, 1) correponds to the delta term for the first neuron in the
    % hidden layer prior to the output node
    % NEED - potential to the neuron, weight matrix (row) that applies to its
    % outputs, DjT values (also a row)
    DiN = zeros(s, n);
    for i = 1:n
        %DiN(s, i) = (Woutput(i, :) .* DjN(s,:)') * (1 - tanh(signals2(s,i))^2);
        
        %Remove the squashing function on the final node
        DiN(s, i) = (Woutput(i, :) * DjN(s,:)');
    end
    
    % Calculate delta terms for each hidden neuron in each stack prior to
    % the final output one
    % DiN = [s x n] deltas for each second hidden layer
    % DiN(1,2) delta for the first neuron in the 2nd stack layer
    for t = s-1:-1:1
        for i = 1:n

            %DiN(t, i) = ((DiN(t+1,1:end-1) * Wprev2(i,:)') + (DjN(t,:) * Woutput(i,:))) * (1 - tanh(signals2(t,i))^2);
            
            DiN(t, i) = ((DiN(t+1,1:end-1) * Wprev2(i,:)') + (DjN(t,:) * Woutput(i,:)'));
        end
    end
    
    DpN = zeros(s, n);
    % Calculate the delta terms for the previous layer
    % DpN = [s x n] deltas for first hidden layer
    % Deltas contributed from same layer in stack above and same stack,
    % next layer
    for i = 1:n
        DpN(s, i) = (DiN(s,1:end-1) * Winterior(i, :)') * (1 - tanh(signals1(s,i))^2);
    end
    
    for t = s-1:-1:1
        for i = 1:n
            DpN(t, i) = ((DpN(t+1,1:end-1) * Wprev1(i,:)') + DiN(t,1:end-1) * Winterior(i,:)') * (1 - tanh(signals1(t,i))^2);
        end
    end
end
