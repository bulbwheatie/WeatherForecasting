% TODO - erroar
function [Winput, Wprev, Woutput] = train(X, Y)
    num_neurons  = 10;
    X = [ones(size(X,1), 1) X]; % Add bias feature

    Winput = initWeights(size(X, 2), num_neurons,-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
    Wprev = initWeights(num_neurons, num_neurons,-1/10, 1/10);
    Woutput = initWeights(num_neurons, size(Y,2), -1/10, 1/10);
    
    iter = 0;
    max_iters = 1000;
    lambda = 0.0000000001;
    while (iter < max_iters)
        i = mod(iter, size(X, 1) - 6) + 1;
        
        %Forward pass through the network with a sequence of training data
        [Ypred, signals] = feedForward(X(i:i+5,:), Winput, Wprev, Woutput);
        
        % Backpropagate and update weight matrices
        [Winput, Wprev, Woutput] = backpropagate(X(i:i+5,:), Y(i:i+5,:), signals, Ypred, Winput, Wprev, Woutput, lambda);       
        
        iter = iter + 1;
    end 
end