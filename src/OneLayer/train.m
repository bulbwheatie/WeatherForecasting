<<<<<<< HEAD
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
=======
% Trains for a specified output feature
function [Winput, Wprev, Woutput, error, train_error] = train(X, outputs, batch_size)
    % Create a small data set and run the entire set through the training
    % process for each update and check that the error is indeed reducing.
    batch_size = 12;
    num_stacks = 3;
    small_set = data;
    
    %Random init of weights
    num_neurons  = 25;
    small_set = [ones(size(small_set,1), 1) small_set]; % Add bias feature
    Winput_init = initWeights(size(small_set, 2), num_neurons,-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
    Winterior_init = initWeights(num_neurons, num_neurons,-1/10, 1/10);
    Wprev1_init = initWeights(num_neurons, num_neurons,-1/10, 1/10);
    Wprev2_init = initWeights(num_neurons, num_neurons,-1/10, 1/10);
    Woutput_init = initWeights(num_neurons, 1, -1/2, 1/2); %Only single output feature in this case
    
    [Winput, Winterior, Wprev1, Wprev2, Woutput, error] = train_new(small_set, Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init, 'temp', batch_size, num_stacks);
    
    i = 1;
    X = data(i:i+11,:);
    X_naive = data(i:i+11,:);
    values_pred = zeros(6,1);
    values_naive= zeros(6,1);
    values_actual = data(i+12:i+17,:);
    
    for j=1:6
        [temp_y, ~, ~] = feedForward_new([ones(size(X,1), 1) X], Winput, Winterior, Wprev1, Wprev2, Woutput);
        values_pred(j,:) = (temp_y(size(temp_y,1),:) .* std_mean(1,2)) + std_mean(2,2);
        values_actual(j,:) = (values_actual(j,:) .* std_mean(1,:)) + std_mean(2,:); %Restore the actual values too
        values_full = data(i+17+j,:);
        values_full(1,2) = values_pred(j,:);
        X = [X(2:size(X,1),:) ; values_full];
        
        [naive_y, ~, ~] = feedForward_new([ones(size(X_naive,1), 1) X_naive], Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init);
        values_naive(j,:) = (naive_y(size(naive_y,1),:) .* std_mean(1,2)) + std_mean(2,2);
        naive_full = data(i+17+j,:);
        naive_full(1,2) = values_naive(j,:);
        X_naive = [X_naive(2:size(X,1),:) ; naive_full];
   
    end
    
    x_axis = 1:size(values_pred, 1);
    
    %temperature -- comapre naive to trained
    plot(x_axis, transpose(values_pred(:,1)), x_axis, transpose(values_naive(:,1)), x_axis, transpose(values_actual(:,2)));
    legend('y = Predicted temperature', 'y = Naive temperature', 'y = Actual temperature','Location','southoutside');
    saveas(gcf, 'temperature2L.fig');
    
    %error
    plot(1:size(error, 1), transpose(error));
    legend('y = Squared error','Location','southeast');
    saveas(gcf, 'train_error2L.fig');
>>>>>>> e35ff221a9aa3847da63799e43846e4446079981
end