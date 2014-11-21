% Trains for a specified output feature wtih an adaptive learning rate

% Inputs:
% data = struct with the following memebers trainX, trainY, validateX,
% validateY, testX, testY (values should be randomized)
% N= number of neurons
% Winput = [d+1 x N] where d+1 is the number of input features + a bias 
% Winterior = Wprev1 = Wprev2 = [N+1 x N] (same size but different values)
% Woutput = [N+1 x l] where l is the nubmer of output features
% batch_size = number of iterations before updating weights
% batches = number of updates to perform

% Outputs
% Weight matrices = same size as respective input corresponding to the
% lowest validation error
% errors = [iter x 1]
function [Winput_min, Winterior_min, Wprev1_min, Wprev2_min, Woutput_min, train_error, valid_error, test_error] = train_BP(data, Winput, Winterior, Wprev1, Wprev2, Woutput, batch_size, batches)

    iter = 1;
    lambda_init = 0.5;
    train_error = zeros(batches, 1);
    valid_error = zeros(batches, 1); 
    diff = 1000;
    valid_sum = 0;
    lookback = 3;
    tmp_error = 0;
    i=1;
    min_error = inf;
    
    % continue until we reach convergence or max iterations
    while (iter <= batches && diff > 10^-5)
        lambda = lambda_init;
        Uinput = zeros(size(Winput));
        Uinterior = zeros(size(Winterior));
        Uprev1 = zeros(size(Wprev1));
        Uprev2 = zeros(size(Wprev2));
        Uoutput = zeros(size(Woutput));
        
        %calculate update weights
        for b=1:batch_size
            if (i == size(data.trainX,3))
               i = 1;
            else 
                i = i + 1;
            end
            [Ypred, signals1, signals1prev, signals2, signals2prev] = feedForward(data.trainX(:,:,i), Winput, Winterior, Wprev1, Wprev2, Woutput);
            [DjN, DiN, DpN] = backpropagate(data.trainX(:,:,i), data.trainY(:,1,i), signals1 + signals1prev, signals2 + signals2prev, Ypred, Winterior, Wprev1, Wprev2, Woutput);       
            [Uinput, Uinterior, Uprev1, Uprev2, Uoutput] = calculateUpdates(Uinput, Uinterior, Uprev1, Uprev2, Uoutput, data.trainX(:,:,i), signals1, signals1prev, signals2, signals2prev, DjN, DiN, DpN);
            tmp_error = tmp_error + sum((Ypred(end,:) - data.trainY(end,1,i)).^2);
        end
        
        train_error(iter) = tmp_error;
        tmp_error = inf;
        
        %Adaptive learning rate
        while (tmp_error > train_error(iter, 1)) && lambda > 10^-8
            disp('adapt');
            tmp_error = 0;
            lambda = lambda/2;
            tmp_Winput = Winput + lambda * Uinput/batch_size;
            tmp_Winterior = Winterior + lambda * Uinterior/batch_size;
            tmp_Wprev1 = Wprev1 + lambda * Uprev1/batch_size;
            tmp_Wprev2 = Wprev2 + lambda * Uprev2/batch_size;
            tmp_Woutput = Woutput + lambda * Uoutput/batch_size;
            
            %Forward pass through the network with validation data to see if error decreases
            for b=1:batch_size
                if (i == size(data.trainX,3))
                   i = 1;
                else 
                    i = i + 1;
                end
                [Ypred, ~, ~, ~, ~] = feedForward(data.trainX(:,:,i), tmp_Winput, tmp_Winterior, tmp_Wprev1, tmp_Wprev2, tmp_Woutput);
                tmp_error = tmp_error + sum((Ypred(end,:) - data.trainY(end,:,i)).^2);
            end
            tmp_error = tmp_error/size(data.trainX, 3);
        end
        
        Winput = tmp_Winput;
        Winterior = tmp_Winterior;
        Wprev1 = tmp_Wprev1;
        Wprev2 = tmp_Wprev2;
        Woutput = tmp_Woutput;
        
        %Calculate validation error
        valid_error_tmp = 0;
        for v=1:size(data.validateX,3)
            [Ypred, ~, ~, ~, ~] = feedForward(data.validateX(:,:,v), Winput, Winterior, Wprev1, Wprev2, Woutput);
            valid_error_tmp = valid_error_tmp + sum((Ypred(end,:) - data.validateY(end,:,v)).^2);
        end
        valid_error_tmp = valid_error_tmp / size(data.validateX,3);
        valid_sum = valid_sum + valid_error_tmp;
        valid_error(iter, 1) = valid_sum/iter;
        if (iter <= lookback + 1)
            diff = 1000;
        else
            diff = abs(valid_error(iter-lookback,1) - valid_sum/iter);
        end 
        
        % Only update the weights if the error decreases
        if valid_error_tmp <= min_error
            Winput_min = Winput;
            Winterior_min = Winterior;
            Wprev1_min = Wprev1;
            Wprev2_min = Wprev2;
            Woutput_min = Woutput;
            min_error = valid_error_tmp;
            disp(min_error);
        end
        iter = iter + 1;
    end
    
    % Comptue the final test error
    test_error = 0;
    for i=1:size(data.testX,3)
        [Ypred, ~, ~, ~, ~] = feedForward(data.testX(:,:,i), Winput_min, Winterior_min, Wprev1_min, Wprev2_min, Woutput_min);
        test_error = test_error + sum((Ypred(end,:) - data.testY(end,:,i)).^2);
    end
    test_error = test_error/size(data.testX,3);
end
