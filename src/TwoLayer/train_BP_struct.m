% Trains for a specified output feature
function [Winput_min, Winterior_min, Wprev1_min, Wprev2_min, Woutput_min, train_error, valid_error, test_error] = train_BP_struct(data, Winput, Winterior, Wprev1, Wprev2, Woutput, batch_size, batches)
    iter = 1;
    max_iters = batch_size*batches;
    lambda = 0.005;
    train_error = zeros(floor(max_iters/(batch_size-1))+1, 1);
    valid_error = zeros(floor(max_iters/(batch_size-1))+1, 1); 
    diff = 1000;
    valid_sum = 10;
    lookback = 1;
    i=0;
    min_error = inf;
    while (iter <= max_iters && diff > 10^-4)
        Uinput = zeros(size(Winput));
        Uinterior = zeros(size(Winterior));
        Uprev1 = zeros(size(Wprev1));
        Uprev2 = zeros(size(Wprev2));
        Uoutput = zeros(size(Woutput));
        tmp_error = 0;
        for b=1:batch_size-1
            if (i == size(data.trainX,3))
               i = 1;
            else 
                i = i + 1;
            end

            %Forward pass through the network with a sequence of training data
            [Ypred, signals1, signals1prev, signals2, signals2prev] = feedForward(data.trainX(:,:,i), Winput, Winterior, Wprev1, Wprev2, Woutput);
            tmp_error = tmp_error + sum((Ypred(end,:) - data.trainY(end,:,i)).^2);
            %disp('Iter:');
            %disp(Ypred(end,:));
            %disp(Y(i+num_stacks-1,:));
            % Backpropagate and update weight matrices
            [DjN, DiN, DpN] = backpropagate(data.trainX(:,:,i), data.trainY(:,:,i), signals1 + signals1prev, signals2 + signals2prev, Ypred, Winterior, Wprev1, Wprev2, Woutput);       
            [Uinput, Uinterior, Uprev1, Uprev2, Uoutput] = calculateUpdates(Uinput, Uinterior, Uprev1, Uprev2, Uoutput, data.trainX(:,:,i), signals1, signals1prev, signals2, signals2prev, DjN, DiN, DpN);
            iter = iter + 1;
        end
        
        %Run the validation data through the network
        valid_error_tmp = 0;
        for v=1:size(data.validateX,3)
            [Ypred, ~, ~, ~, ~] = feedForward(data.validateX(:,:,v), Winput, Winterior, Wprev1, Wprev2, Woutput);
            valid_error_tmp = valid_error_tmp + sum((Ypred(end,:) - data.validateY(end,:,v)).^2);
        end
        valid_sum = valid_sum + valid_error_tmp;
        valid_error_tmp = valid_error_tmp / size(data.validateX,3);
                
        % If the error is better, then store the weights
        if valid_error_tmp <= min_error
            Winput_min = Winput;
            Winterior_min = Winterior;
            Wprev1_min = Wprev1;
            Wprev2_min = Wprev2;
            Woutput_min = Woutput;
            min_error = valid_error_tmp;
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
            diff = abs(valid_error(floor(iter/(batch_size-1))-lookback,1) - valid_sum/iter);
        end        
    end
    test_error = 0;
    for i=1:size(data.testX,3)
        [Ypred, ~, ~, ~, ~] = feedForward(data.testX(:,:,i), Winput_min, Winterior_min, Wprev1_min, Wprev2_min, Woutput_min);
        test_error = test_error + sum((Ypred(end,:) - data.testY(end,:,i)).^2);
    end
    test_error = test_error/size(data.testX,3);
end
