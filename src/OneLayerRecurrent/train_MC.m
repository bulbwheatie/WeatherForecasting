% Trains for a specified output feature
function [Winput, Wprev, Woutput, error] = train_random(X, Winput, Wprev, Woutput, mode, num_stacks, max_epochs)
    if (strcmp(mode, 'temp') == 1)
        % Only train against the 2nd column of the outputs
        Y = X(2:size(X), 2);
    else
        Y = X(2:size(X), 2);
    end

    X = X(1:size(X)-1, :);
    denom = 10;
    count_limit = 10;
    epoch = 1;
    batch_size = size(X, 1) - num_stacks + 1;
    error = zeros(max_epochs, 1);
        
    for i=1:batch_size
        %Forward pass through the network with a sequence of training data
        [Ypred, signals] = feedForward(X(i:i+num_stacks-1,:), Winput, Wprev, Woutput);
        error(1, 1) = error(1, 1) + sum((Ypred(size(Ypred, 1),:) - Y(i+num_stacks-1,:)).^2);
    end
    error(1,1) = error(1,1)/batch_size;
    while (epoch <= max_epochs)
        disp('Starting epoch...');
        disp(epoch);
        disp(error(epoch, 1));
        
        %set Winput
        tmp_error = inf;
        count = 1;
        while tmp_error > error(epoch, 1)
            if count > count_limit
                count = 1;
                denom = denom * 2;
            end
            disp('setting Winput');
            disp(epoch);
            disp(tmp_error);
            disp(error(epoch,1));
            tmp_Winput = Winput + initWeights(size(Winput, 1), size(Winput, 2), -2/denom, 2/denom);
            tmp_error = 0;
            for i=1:batch_size
                %Forward pass through the network with a sequence of training data
                [Ypred, signals] = feedForward(X(i:i+num_stacks-1,:), tmp_Winput, Wprev, Woutput);
                tmp_error = tmp_error + sum((Ypred(size(Ypred, 1),:) - Y(i+num_stacks-1,:)).^2);
            end
            tmp_error = tmp_error/batch_size;
            count = count+1;
        end
        Winput = tmp_Winput;
        error(epoch, 1) = tmp_error;
        
        %set Wprev
        tmp_error = inf;
        count = 1;
        while tmp_error > error(epoch, 1)
            if count > count_limit
                count = 1;
                denom = denom * 2;
            end
            disp('setting Wprev');
            disp(epoch);
            disp(tmp_error);
            disp(error(epoch,1));
            tmp_Wprev = Wprev + initWeights(size(Wprev, 1), size(Wprev, 2), -2/denom, 2/denom);
            tmp_error = 0;
            for i=1:batch_size
                %Forward pass through the network with a sequence of training data
                [Ypred, signals] = feedForward(X(i:i+num_stacks-1,:), Winput, tmp_Wprev, Woutput);
                tmp_error = tmp_error + sum((Ypred(size(Ypred, 1),:) - Y(i+num_stacks-1,:)).^2);
            end
            tmp_error = tmp_error/batch_size;
            count = count+1;
        end
        Wprev = tmp_Wprev;
        error(epoch, 1) = tmp_error;
        
        %set Woutput
        tmp_error = inf;
        count = 1;
        while tmp_error > error(epoch, 1)
            if count > count_limit
                count = 1;
                denom = denom * 2;
            end
            disp('setting Woutput');
            disp(epoch);
            disp(tmp_error);
            disp(error(epoch,1));
            tmp_Woutput = Woutput + initWeights(size(Woutput, 1), size(Woutput, 2), -3/denom, 3/denom);
            tmp_error = 0;
            for i=1:batch_size
                %Forward pass through the network with a sequence of training data
                [Ypred, signals] = feedForward(X(i:i+num_stacks-1,:), Winput, Wprev, tmp_Woutput);
                tmp_error = tmp_error + sum((Ypred(size(Ypred, 1),:) - Y(i+num_stacks-1,:)).^2);
            end
            tmp_error = tmp_error/batch_size;
            count = count+1;
        end
        Woutput = tmp_Woutput;
        error(epoch, 1) = tmp_error;
        
        if (epoch < size(error, 1))
            error(epoch+1, 1) = tmp_error;
        end
        epoch = epoch + 1;
    end 