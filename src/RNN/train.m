% TODO - erroar
function [Winput, Wprev, Woutput, error] = train(X)
    Y = X(2:size(X), :);
    X = X(1:size(X)-1, :);
    num_neurons  = 10;
    X = [ones(size(X,1), 1) X]; % Add bias feature

    Winput = initWeights(size(X, 2), num_neurons,-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
    Wprev = initWeights(num_neurons, num_neurons,-1/10, 1/10);
    Woutput = initWeights(num_neurons, size(Y,2), -1/10, 1/10);
    
    iter = 1;
    max_iters = 2000;
    lambda = 0.000000000000000001;
    error = zeros(max_iters, 1);
    while (iter <= max_iters)
        i = mod(iter, size(X, 1) - 12) + 1;
        
        %Forward pass through the network with a sequence of training data
        [Ypred, signals] = feedForward(X(i:i+11,:), Winput, Wprev, Woutput);
        error(iter, 1) = sum((Ypred(size(Ypred,1),:) - Y(i+11,:)).^2);
        % Backpropagate and update weight matrices
        [Winput, Wprev, Woutput] = backpropagate(X(i:i+11,:), Y(i:i+11,:), signals, Ypred, Winput, Wprev, Woutput, lambda);       
        
        iter = iter + 1;
    end 
end