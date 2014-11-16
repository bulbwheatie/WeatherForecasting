% Trains for a specified output feature
function [Winput, Wprev, Woutput, error, train_error] = train(X, outputs, batch_size)
    if (strcmp(outputs, 'temp') == 1)
        % Only train against the 2nd column of the outputs
        Y = X(2:size(X), 2);        
    else
        Y = X(2:size(X), :);        
    end

    % Hyperparameters 
    num_neurons  = 30;
    max_iters = 3000;
    lambda = 0.00000000001;
    
    %Preprocess input samples
    X = X(1:size(X)-1, :);
    X = [ones(size(X,1), 1) X]; % Add bias feature
    
    %Random Initial weights
    Winput = initWeights(size(X, 2), num_neurons,-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
    Wprev = initWeights(num_neurons, num_neurons,-1/10, 1/10);
    Woutput = initWeights(num_neurons, size(Y,2), -1/3, 1/3);
    
    iter = 0;
    num_stacks = 12;
    batch_size = batch_size + num_stacks;
    error = zeros(floor(max_iters/batch_size), 1);
    train_error = zeros(floor(max_iters/batch_size), 1);

    while (iter <= max_iters)
        Uinput = zeros(size(Winput));
        Uprev = zeros(size(Wprev));
        Uoutput = zeros(size(Woutput));
        tmp_error = 0;
        for b=1:batch_size-(2* num_stacks)
            i = mod(iter, size(X, 1) - (num_stacks -1)) + 1;

            %Forward pass through the network with a sequence of training data
            [Ypred, signals] = feedForward(X(i:i+num_stacks-1,:), Winput, Wprev, Woutput);
            tmp_error = tmp_error + sum((Ypred(size(Ypred,1),:) - Y(i+num_stacks-1,:)).^2);
            
            % Backpropagate and update weight matrices
            [DiN] = backpropagate(X(i:i+num_stacks-1,:), Y(i:i+num_stacks-1,:), signals, Ypred, Wprev, Woutput);       
            [Uinput, Uprev, Uoutput] = calculateUpdates(Uinput, Uoutput, Uprev, X, signals, DiN);
            iter = iter + 1;
        end
        % Update the weight matrices based on average deltas
        Winput = Winput + lambda * Uinput/batch_size;
        Wprev = Wprev + lambda * Uprev/batch_size;
        Woutput = Woutput + lambda * Uoutput/batch_size;
        train_error(floor(iter/batch_size) + 1, 1)= tmp_error/batch_size;
        train_error(floor(iter/batch_size) + 1, 1)
    end 
end