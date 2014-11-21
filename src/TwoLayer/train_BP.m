% Trains for a specified output feature based on an in order data sequence
% USED to validate backpropagation functions; should not be used for actual
% training

% Inputs:
% data = [m x d] where m is number of samples and d is features with bias
% applied
% N= number of neurons
% Winput = [d+1 x N] where d+1 is the number of input features + a bias 
% Winterior = Wprev1 = Wprev2 = [N+1 x N] (same size but different values)
% Woutput = [N+1 x l] where l is the nubmer of output features
% batch_size = number of iterations before updating weights
% num_stacks = number of stacks in the network

% Outputs
% Weight matrices = same size as respective input corresponding to the
% lowest validation error
% errors = [iter x 1]
function [Winput_min, Winterior_min, Wprev1_min, Wprev2_min, Woutput_min, train_error, valid_error] = train_BP(X, Xvalid, Winput, Winterior, Wprev1, Wprev2, Woutput, mode, batch_size, num_stacks)
    if (strcmp(mode, 'temp') == 1)
        % Only train against the 2nd column of the outputs
        Y = X(2:size(X), 2);
        Yvalid = Xvalid(2:size(Xvalid),2);
    else
        Y = X(2:size(X), :);
        Yvalid = Xvalid(2:size(Xvalid), :);
    end

    X = X(1:size(X)-1, :);
    Xvalid = Xvalid(1:size(Xvalid)-1, :);
    iter = 1;
    max_iters = batch_size*500;
    lambda = 0.0001;
    train_error = zeros(floor(max_iters/batch_size), 1);
    valid_error = zeros(floor(max_iters/batch_size), 1); %Store interals with the same value
    diff = 1000;
    valid_sum = 0;
    lookback = 1;
    i=0;
    while (iter <= max_iters)
        Uinput = zeros(size(Winput));
        Uinterior = zeros(size(Winterior));
        Uprev1 = zeros(size(Wprev1));
        Uprev2 = zeros(size(Wprev2));
        Uoutput = zeros(size(Woutput));
        tmp_error = 0;
        for b=1:batch_size-1
            if (i == size(X,1) - num_stacks)
               i = 1;
            else 
                i = i + 1;
            end

            %Forward pass through the network with a sequence of training data
            [Ypred, signals1, signals1prev, signals2, signals2prev] = feedForward(X(i:i+num_stacks-1,:), Winput, Winterior, Wprev1, Wprev2, Woutput);
            tmp_error = tmp_error + sum((Ypred(end,:) - Y(i+num_stacks-1,:)).^2);
            %disp('Iter:');
            %disp(Ypred(end,:));
            %disp(Y(i+num_stacks-1,:));
            % Backpropagate and update weight matrices
            [DjN, DiN, DpN] = backpropagate(X(i:i+num_stacks-1,:), Y(i:i+num_stacks-1,:), signals1 + signals1prev, signals2 + signals2prev, Ypred, Winterior, Wprev1, Wprev2, Woutput);       
            [Uinput, Uinterior, Uprev1, Uprev2, Uoutput] = calculateUpdates(Uinput, Uinterior, Uprev1, Uprev2, Uoutput, X(i:i+num_stacks-1,:), signals1, signals1prev, signals2, signals2prev, DjN, DiN, DpN);
            iter = iter + 1;
        end
        
        %Run the validation data through the network
        for v=1:size(Xvalid,1) - num_stacks+1
            [Ypred, ~, ~, ~, ~] = feedForward(Xvalid(v:v+num_stacks-1,:), Winput, Winterior, Wprev1, Wprev2, Woutput);
            valid_sum = valid_sum + sum((Ypred(end,:) - Yvalid(v+num_stacks-1,:)).^2);
        end        
                
        % If the error is better, then store the weights
        if (floor(iter/(batch_size-1)) == 1 || valid_sum/iter <= min_error ) 
            Winput_min = Winput;
            Winterior_min = Winterior;
            Wprev1_min = Wprev1;
            Wprev2_min = Wprev2;
            Woutput_min = Woutput;
            min_error = valid_sum/iter;
            disp(min_error);
        end
        
        % Update the weight matrices based on average deltas
        Winput = Winput + lambda * Uinput/(batch_size-1);
        Winterior = Winterior + lambda * Uinterior/(batch_size-1);
        Wprev1 = Wprev1 + lambda * Uprev1/(batch_size-1);
        Wprev2 = Wprev2 + lambda * Uprev2/(batch_size-1);
        Woutput = Woutput + lambda * Uoutput/(batch_size-1);
        train_error(floor(iter/(batch_size-1)), 1) = tmp_error/(batch_size-1);
        
        %Update validation error
        valid_error(floor(iter/(batch_size-1)), 1) = valid_sum/iter;
        if (floor(iter/(batch_size-1)) <= lookback)
            diff = 1000;
        else
            diff = abs(valid_error(floor(iter/(batch_size-1))-lookback) - valid_sum/iter);
        end        
    end 
end
