% TODO - erroar
% Trains for a specified output feature
function [Winput, Winterior, Wprev1, Wprev2, Woutput, error] = train_BP_adaptive(data, Winput, Winterior, Wprev1, Wprev2, Woutput, max_epochs)

    error = zeros(max_epochs, 1);
    lambda_init = 0.01;
    num_samples = size(data.trainX, 3);
    %Forward pass through the network with a sequence of training data
    [Ypred, signals1, signals1prev, signals2, signals2prev] = feedForward_struct(data.trainX, Winput, Winterior, Wprev1, Wprev2, Woutput);
    error(1,1) = sum(sum((Ypred(size(data.trainY, 1),:,:) - data.trainY(size(data.trainY, 1),:,:)).^2))/num_samples;
    
    for epoch=1:max_epochs
        lambda = lambda_init;
        disp('Starting epoch...');
        disp(epoch);
        disp(error(epoch, 1));

        Uinput = zeros(size(Winput));
        Uinterior = zeros(size(Winterior));
        Uprev1 = zeros(size(Wprev1));
        Uprev2 = zeros(size(Wprev2));
        Uoutput = zeros(size(Woutput));
        
        %calculate update weights
        for row=1:num_samples
            [DjN, DiN, DpN] = backpropagate(data.trainX(:,:,row), data.trainY(:,:,row), signals1 + signals1prev, signals2 + signals2prev, Ypred(:,:,row), Winterior, Wprev1, Wprev2, Woutput);       
            [Uinput, Uinterior, Uprev1, Uprev2, Uoutput] = calculateUpdates(Uinput, Uinterior, Uprev1, Uprev2, Uoutput, data.trainX(:,:,row), signals1, signals1prev, signals2, signals2prev, DjN, DiN, DpN);
        end
        
        %adaptive learning rate
        %tmp_error = inf;
        %while tmp_error > error(epoch, 1)
        %    disp('starting while loop');
        %    disp(epoch);
        %    disp(tmp_error);
       %     disp(error(epoch,1));
       %     disp(lambda);
            tmp_Winput = Winput + lambda * Uinput/num_samples;
            tmp_Winterior = Winterior + lambda * Uinterior/num_samples;
            tmp_Wprev1 = Wprev1 + lambda * Uprev1/num_samples;
            tmp_Wprev2 = Wprev2 + lambda * Uprev2/num_samples;
            tmp_Woutput = Woutput + lambda * Uoutput/num_samples;
            
            %Forward pass through the network with a sequence of training data
            [Ypred, signals1, signals1prev, signals2, signals2prev] = feedForward_struct(data.trainX, Winput, Winterior, Wprev1, Wprev2, Woutput);
            tmp_error = sum(sum((Ypred(size(data.trainY, 1),:,:) - data.trainY(size(data.trainY, 1),:,:)).^2))/num_samples;
            
        %    lambda = lambda/2;
        %end
        
        Winput = tmp_Winput;
        Winterior = tmp_Winterior;
        Wprev1 = tmp_Wprev1;
        Wprev2 = tmp_Wprev2;
        Woutput = tmp_Woutput;
        
        error(epoch, 1) = tmp_error;
        if (epoch < size(error, 1))
            error(epoch+1, 1) = tmp_error;
        end
    end
end