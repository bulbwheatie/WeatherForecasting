
function [Winput, Wprev, Woutput, error] = train(X, Y)
    addpath('..');
    
    X = [ones(size(X,1), 1) X]; % Add bias feature

    Winput = init_weights(size(X, 1), size(X, 2),-1/2, 1/2); % Create a d + 1 x n matrix for the extra bias feature
    Wprev = init_weights(size(X, 2), size(X, 2),-1/2, 1/2);
    Woutput = init_weights(size(X, 2), num_output, -1/2, 1/2);
    
    iter = 0;
    while (iter < max_iters)
        i = mod(iter, size(X, 1) - 24);
        
        %Forward pass through the network with a sequence of training data
        [Ypred, signals] = feedforward(X(i:i+24,:), num_outputs, Winput, Wprev, Woutput);
        
        % Backpropagate and update weight matrices
        [Winput, Wprev, Woutput] = backpropagate(X, Y, signals, Ypred, Winput, Wprev, Woutput);
        
        
        iter = iter + 1;
    end 
end