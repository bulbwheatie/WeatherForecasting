% TODO - erroar
% Trains for a specified output feature
function [Winput, Wprev, Woutput, error] = train(X, outputs, batch_size)
    if (strcmp(outputs, 'temp') == 1)
        % Only train against the 2nd column of the outputs
        Y = X(2:size(X), 2);
    else
        Y = X(2:size(X), :);
    end

    X = X(1:size(X)-1, :);
    num_neurons  = 30;
    X = [ones(size(X,1), 1) X]; % Add bias feature

    Winput = initWeights(size(X, 2), num_neurons,-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
    Wprev = initWeights(num_neurons, num_neurons,-1/10, 1/10);
    Woutput = initWeights(num_neurons, size(Y,2), -1/2, 1/2);
    
    iter = 1;
    batch_size = 1000;
    max_iters = batch_size*50;
    lambda = 0.0000000000000000000001;
    error = zeros(floor(max_iters/batch_size), 1);
    while (iter <= max_iters)
        Uinput = zeros(size(Winput));
        Uprev = zeros(size(Wprev));
        Uoutput = zeros(size(Woutput));
        tmp_error = 0;
        for b=1:batch_size
            i = mod(iter, size(X, 1) - 12) + 1;

            %Forward pass through the network with a sequence of training data
            [Ypred, signals] = feedForward(X(i:i+11,:), Winput, Wprev, Woutput);
            %disp('Ypred');
            %disp(Ypred);
            tmp_error = tmp_error + sum((Ypred(size(Ypred,1),:) - Y(i+11,:)).^2);
            
            % Backpropagate and update weight matrices
            [DiN] = backpropagate(X(i:i+11,:), Y(i:i+11,:), signals, Ypred, Wprev, Woutput);       
            [Uinput, Uprev, Uoutput] = calculateUpdates(Uinput, Uoutput, Uprev, X, signals, DiN);
            iter = iter + 1;
        end
        % Update the weight matrices based on average deltas
        Winput = Winput + lambda * Uinput/batch_size;
        Wprev = Wprev + lambda * Uprev/batch_size;
        Woutput = Woutput + lambda * Uoutput/batch_size;
        error(floor(iter/batch_size), 1) = tmp_error/batch_size;
        
        %disp('error');
        %disp(error(floor(iter/batch_size), 1));
        %disp('iter');
        disp(iter-1);
    end 
end