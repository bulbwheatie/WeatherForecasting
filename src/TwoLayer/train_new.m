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
    
    epoch = 1;
    batch_size = size(X, 1) - num_stacks + 1;
    max_epochs = 50;
    error = zeros(max_epochs, 1);
        
    for i=1:batch_size
        %Forward pass through the network with a sequence of training data
        [Ypred, signals1, signals2] = feedForward_new(X(i:i+num_stacks-1,2), Winput, Winterior, Wprev1, Wprev2, Woutput);
        error(1, 1) = error(1, 1) + sum((Ypred(size(Ypred,1),:) - Y(i+num_stacks-1,:)).^2);
    end
    
    while (epoch <= max_epochs)
        lambda = 10^-1;
        disp('Starting epoch...');
        disp(epoch);
        disp(error(epoch, 1));

        Uinput = zeros(size(Winput));
        Uinterior = zeros(size(Winterior));
        Uprev1 = zeros(size(Wprev1));
        Uprev2 = zeros(size(Wprev2));
        Uoutput = zeros(size(Woutput));
        
        %calculate update weights
        for i=1:batch_size
            [DiN, DpN] = backpropagate_new(X(i:i+num_stacks-1,:), Y(i:i+num_stacks-1,:), signals1, signals2, Ypred, Winterior, Wprev1, Wprev2, Woutput);       
            [Uinput, Uinterior, Uprev1, Uprev2, Uoutput] = calculateUpdates_new(Uinput, Uinterior, Uprev1, Uprev2, Uoutput, X, signals1, signals2, DiN, DpN);
        end
        
        %adaptive learning rate
        tmp_error = inf;
        while tmp_error > error(epoch, 1)
            disp('starting while loop');
            disp(epoch);
            disp(tmp_error);
            disp(error(epoch,1));
            disp(lambda);
            tmp_Winput = Winput + lambda * Uinput/batch_size;
            tmp_Winterior = Winterior + lambda * Uinterior/batch_size;
            tmp_Wprev1 = Wprev1 + lambda * Uprev1/batch_size;
            tmp_Wprev2 = Wprev2 + lambda * Uprev2/batch_size;
            tmp_Woutput = Woutput + lambda * Uoutput/batch_size;
            tmp_error = 0;
            for i=1:batch_size
                %Forward pass through the network with a sequence of training data
                [Ypred, signals1, signals2] = feedForward_new(X(i:i+num_stacks-1,2), tmp_Winput, tmp_Winterior, tmp_Wprev1, tmp_Wprev2, tmp_Woutput);
                tmp_error = tmp_error + sum((Ypred(size(Ypred,1),:) - Y(i+num_stacks-1,:)).^2);
            end
            lambda = lambda/10;
        end
        
        Winput = tmp_Winput;
        Winterior = tmp_Winterior;
        Wprev1 = tmp_Wprev1;
        Wprev2 = tmp_Wprev2;
        Woutput = tmp_Woutput;
        
        error(epoch, 1) = tmp_error;
        if (epoch < size(error, 1))
            error(epoch+1, 1) = tmp_error;
        end
        epoch = epoch + 1;
    end 
end