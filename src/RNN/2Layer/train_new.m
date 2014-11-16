% TODO - erroar
% Trains for a specified output feature
function [Winput, Winterior, Wprev1, Wprev2, Woutput, error] = train_new(X, Winput, Winterior, Wprev1, Wprev2, Woutput, mode, batch_size, num_stacks)
    if (strcmp(mode, 'temp') == 1)
        % Only train against the 2nd column of the outputs
        Y = X(2:size(X), 2);
    else
        Y = X(2:size(X), :);
    end

    X = X(1:size(X)-1, :);
    
    iter = 1;
    max_iters = batch_size*200;
    lambda = 0.0000000000001;
    error = zeros(floor(max_iters/batch_size), 1);
    while (iter <= max_iters)
        Uinput = zeros(size(Winput));
        Uinterior = zeros(size(Winterior));
        Uprev1 = zeros(size(Wprev1));
        Uprev2 = zeros(size(Wprev2));
        Uoutput = zeros(size(Woutput));
        tmp_error = 0;
        for b=1:batch_size-1
            i = mod(iter, size(X, 1) - num_stacks) + 1;

            %Forward pass through the network with a sequence of training data
            [Ypred, signals1, signals2] = feedForward_new(X(i:i+num_stacks-1,2), Winput, Winterior, Wprev1, Wprev2, Woutput);
            tmp_error = tmp_error + sum((Ypred(size(Ypred,1),:) - Y(i+num_stacks-1,:)).^2);
            
            % Backpropagate and update weight matrices
            [DiN, DpN] = backpropagate_new(X(i:i+num_stacks-1,:), Y(i:i+num_stacks-1,:), signals1, signals2, Ypred, Winterior, Wprev1, Wprev2, Woutput);       
            [Uinput, Uinterior, Uprev1, Uprev2, Uoutput] = calculateUpdates_new(Uinput, Uinterior, Uprev1, Uprev2, Uoutput, X, signals1, signals2, DiN, DpN);
            iter = iter + 1;
        end
        % Update the weight matrices based on average deltas
        Winput = Winput + lambda * Uinput/batch_size;
        Winterior = Winterior + lambda * Uinterior/batch_size;
        Wprev1 = Wprev1 + lambda * Uprev1/batch_size;
        Wprev2 = Wprev2 + lambda * Uprev2/batch_size;
        Woutput = Woutput + lambda * Uoutput/batch_size;
        error(floor(iter/batch_size), 1) = tmp_error/batch_size;
        disp(error(floor(iter/batch_size), 1));
    end 
end