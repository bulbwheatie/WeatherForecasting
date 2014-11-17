% Trains for a specified output feature
function [Winput, Winterior, Wprev1, Wprev2, Woutput, error] = train_random_new(X, Winput, Winterior, Wprev1, Wprev2, Woutput, num_stacks, max_epochs)
    Y = X(2:size(X,1), 2);
    X = X(1:size(X,1)-1, :);
    Winput_denom = 30;
    Winterior_denom = 30;
    Wprev1_denom = 30;
    Wprev2_denom = 30;
    Woutput_denom = 2.5;
    count_limit = 5;
    epoch = 1;
    batch_size = size(X, 1) - num_stacks + 1;
    error = zeros(max_epochs, 1);
        
    for i=1:batch_size
        %Forward pass through the network with a sequence of training data
        [Ypred, signals1, signals2] = feedForward_new(X(i:i+num_stacks-1,:), Winput, Winterior, Wprev1, Wprev2, Woutput);
        error(1, 1) = error(1, 1) + sum((Ypred(size(Ypred, 1),:) - Y(i+num_stacks-1,:)).^2);
    end
    error(1,1) = error(1,1)/batch_size;
    while (epoch <= max_epochs)
        %set Winput
        tmp_error = inf;
        count = 1;
        while tmp_error > error(epoch, 1) && Winput_denom < 100
            if count > count_limit
                count = 1;
                Winput_denom = Winput_denom * 2;
                tmp_Winput = Winput;
                tmp_error = error(epoch, 1);
            else
                disp('setting Winput');
                disp(epoch);
                disp(tmp_error);
                disp(error(epoch,1));
                disp(Winput_denom);
                tmp_Winput = Winput + initWeights(size(Winput, 1), size(Winput, 2), -1/Winput_denom, 1/Winput_denom);
                tmp_error = 0;
                for i=1:batch_size
                    %Forward pass through the network with a sequence of training data
                    [Ypred, signals1, signals2] = feedForward_new(X(i:i+num_stacks-1,:), tmp_Winput, Winterior, Wprev1, Wprev2, Woutput);
                    tmp_error = tmp_error + sum((Ypred(size(Ypred, 1),:) - Y(i+num_stacks-1,:)).^2);
                end
                tmp_error = tmp_error/batch_size;
                count = count+1;
            end
        end
        Winput = tmp_Winput;
        error(epoch, 1) = tmp_error;
        
        %set Winterior
        tmp_error = inf;
        count = 1;
        while tmp_error > error(epoch, 1) && Winterior_denom < 100
            if count > count_limit
                count = 1;
                Winterior_denom = Winterior_denom * 2;
                tmp_Winterior = Winterior;
                tmp_error = error(epoch, 1);
            else
                disp('setting Winterior');
                disp(epoch);
                disp(tmp_error);
                disp(error(epoch,1));
                disp(Winterior_denom);
                tmp_Winterior = Winterior + initWeights(size(Winterior, 1), size(Winterior, 2), -1/Winterior_denom, 1/Winterior_denom);
                tmp_error = 0;
                for i=1:batch_size
                    %Forward pass through the network with a sequence of training data
                    [Ypred, signals1, signals2] = feedForward_new(X(i:i+num_stacks-1,:), Winput, tmp_Winterior, Wprev1, Wprev2, Woutput);
                    tmp_error = tmp_error + sum((Ypred(size(Ypred, 1),:) - Y(i+num_stacks-1,:)).^2);
                end
                tmp_error = tmp_error/batch_size;
                count = count+1;
            end
        end
        Winterior = tmp_Winterior;
        error(epoch, 1) = tmp_error;
        
        %set Wprev1
        tmp_error = inf;
        count = 1;
        while tmp_error > error(epoch, 1) && Wprev1_denom < 100
            if count > count_limit
                count = 1;
                Wprev1_denom = Wprev1_denom * 2;
                tmp_Wprev1 = Wprev1;
                tmp_error = error(epoch, 1);
            else
                disp('setting Wprev1');
                disp(epoch);
                disp(tmp_error);
                disp(error(epoch,1));
                disp(Wprev1_denom);
                tmp_Wprev1 = Wprev1 + initWeights(size(Wprev1, 1), size(Wprev1, 2), -1/Wprev1_denom, 1/Wprev1_denom);
                tmp_error = 0;
                for i=1:batch_size
                    %Forward pass through the network with a sequence of training data
                    [Ypred, signals1, signals2] = feedForward_new(X(i:i+num_stacks-1,:), Winput, Winterior, tmp_Wprev1, Wprev2, Woutput);
                    tmp_error = tmp_error + sum((Ypred(size(Ypred, 1),:) - Y(i+num_stacks-1,:)).^2);
                end
                tmp_error = tmp_error/batch_size;
                count = count+1;
            end
        end
        Wprev1 = tmp_Wprev1;
        error(epoch, 1) = tmp_error;
        
        %set Wprev2
        tmp_error = inf;
        count = 1;
        while tmp_error > error(epoch, 1) && Wprev2_denom < 100
            if count > count_limit
                count = 1;
                Wprev2_denom = Wprev2_denom * 2;
                tmp_Wprev2 = Wprev2;
                tmp_error = error(epoch, 1);
            else
                disp('setting Wprev2');
                disp(epoch);
                disp(tmp_error);
                disp(error(epoch,1));
                disp(Wprev2_denom);
                tmp_Wprev2 = Wprev2 + initWeights(size(Wprev2, 1), size(Wprev2, 2), -1/Wprev2_denom, 1/Wprev2_denom);
                tmp_error = 0;
                for i=1:batch_size
                    %Forward pass through the network with a sequence of training data
                    [Ypred, signals1, signals2] = feedForward_new(X(i:i+num_stacks-1,:), Winput, Winterior, Wprev1, tmp_Wprev2, Woutput);
                    tmp_error = tmp_error + sum((Ypred(size(Ypred, 1),:) - Y(i+num_stacks-1,:)).^2);
                end
                tmp_error = tmp_error/batch_size;
                count = count+1;
            end
        end
        Wprev2 = tmp_Wprev2;
        error(epoch, 1) = tmp_error;
        
        %set Woutput
        tmp_error = inf;
        count = 1;
        while tmp_error > error(epoch, 1) && Woutput_denom < 100
            if count > count_limit
                count = 1;
                Woutput_denom = Woutput_denom * 2;
                tmp_Woutput = Woutput;
                tmp_error = error(epoch, 1);
            else
                disp('setting Woutput');
                disp(epoch);
                disp(tmp_error);
                disp(error(epoch,1));
                disp(Woutput_denom);
                tmp_Woutput = Woutput + initWeights(size(Woutput, 1), size(Woutput, 2), -1/Woutput_denom, 1/Woutput_denom);
                tmp_error = 0;
                for i=1:batch_size
                    %Forward pass through the network with a sequence of training data
                    [Ypred, signals1, signals2] = feedForward_new(X(i:i+num_stacks-1,:), Winput, Winterior, Wprev1, Wprev2, tmp_Woutput);
                    tmp_error = tmp_error + sum((Ypred(size(Ypred, 1),:) - Y(i+num_stacks-1,:)).^2);
                end
                tmp_error = tmp_error/batch_size;
                count = count+1;
            end
        end
        Woutput = tmp_Woutput;
        error(epoch, 1) = tmp_error;
        
        if (epoch < size(error, 1))
            error(epoch+1, 1) = tmp_error;
        end
        epoch = epoch + 1;
    end
end 
    