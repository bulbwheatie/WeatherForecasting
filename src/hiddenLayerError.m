% Computes the delta for hidden layers in the network

% Inputs
% deltaPrev = m x d delta matrix for the previous layer
% W = n x p weight matrix of previous layer where n is number of input features and p is
% number of output features
% Xout = signals leaving this layer

% Outputs
% delta = m x d where m is number of samples and d is number of perceptrons



function delta = hiddenLayerError(deltaPrev, W, Xout)
    
    delta = zeros(size(Xout,1), 1);
    for i = 1:(size(W, 1))
        sum = 0;
        for j = 1:(size(deltaPrev, 2))
            sum = sum + deltaPrev(i ,j) * W(i, j);
        end
        % delta(i, :) = sum * (1 - Xout(1,i) * Xout(1,i));
        delta(i, 1) = -sum;
    end
end