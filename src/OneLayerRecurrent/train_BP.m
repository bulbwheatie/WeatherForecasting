% Trains for a specified output feature
function [Winput, Wprev, Woutput, error] = train_BP(X, Winput, Wprev, Woutput, mode, num_stacks, max_epochs)
    if (strcmp(mode, 'temp') == 1)
        % Only train against the 2nd column of the outputs
        Y = X(2:size(X), 2);
    else
        Y = X(2:size(X), 2);
    end

    X = X(1:size(X)-1, :);
    
    epoch = 1;
    batch_size = size(X, 1) - num_stacks + 1;
    error = zeros(max_epochs, 1);
        
    for i=1:batch_size
        %Forward pass through the network with a sequence of training data
        [Ypred, signals] = feedForward(X(i:i+num_stacks-1,:), Winput, Wprev, Woutput);
        error(1, 1) = error(1, 1) + sum((Ypred(size(Ypred,1),:) - Y(i+num_stacks-1,:)).^2);
    end
    error(1,1) = error(1,1)/batch_size;
    
    while (epoch <= max_epochs)
        lambda = 0.1;
        disp('Starting epoch...');
        disp(epoch);
        disp(error(epoch, 1));

        Uinput = zeros(size(Winput));
        Uprev = zeros(size(Wprev));
        Uoutput = zeros(size(Woutput));
        
        %calculate update weights
        for i=1:batch_size
            DiN = backpropagate(X(i:i+num_stacks-1,:), Y(i:i+num_stacks-1,:), signals, Ypred, Wprev, Woutput);       
            [Uinput, Uprev, Uoutput] = calculateUpdates(Uinput, Uprev, Uoutput, X, signals, DiN);
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
            tmp_Wprev = Wprev + lambda * Uprev/batch_size;
            tmp_Woutput = Woutput + lambda * Uoutput/batch_size;
            tmp_error = 0;
            for i=1:batch_size
                %Forward pass through the network with a sequence of training data
                [Ypred, signals] = feedForward(X(i:i+num_stacks-1,:), tmp_Winput, tmp_Wprev, tmp_Woutput);
                tmp_error = tmp_error + sum((Ypred - Y(i+num_stacks-1,:)).^2);
            end
            tmp_error = tmp_error/batch_size;
            lambda = lambda/10;
        end
        
        Winput = tmp_Winput;
        Wprev = tmp_Wprev;
        Woutput = tmp_Woutput;
        
        error(epoch, 1) = tmp_error;
        if (epoch < size(error, 1))
            error(epoch+1, 1) = tmp_error;
        end
        epoch = epoch + 1;
    end 