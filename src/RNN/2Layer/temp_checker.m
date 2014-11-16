% Models the temperature output based on all features

function temp_checker(data, std_mean)
    batch_size = 20;
    num_stacks = 12;
    
    %Random init of weights
    num_neurons  = 25;
    X = [ones(size(data,1), 1) data]; % Add bias feature
    Winput_init = initWeights(size(X, 2), num_neurons,-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
    Winterior_init = initWeights(num_neurons, num_neurons,-1/10, 1/10);
    Wprev1_init = initWeights(num_neurons, num_neurons,-1/10, 1/10);
    Wprev2_init = initWeights(num_neurons, num_neurons,-1/10, 1/10);
    Woutput_init = initWeights(num_neurons, 1, -1/2, 1/2); %Only single output feature in this case
    
    [Winput, Winterior, Wprev1, Wprev2, Woutput, error] = train_new(X, Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init, 'temp', batch_size, num_stacks);
    
    i = 1;
    X = data(i:i+11,:);
    values_pred = zeros(6,1);
    values_actual = data(i+12:i+17,:);
    X_naive = data(i:i+11,:);
    values_naive= zeros(6,1);

    
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
    
    %temperature
    plot(x_axis, transpose(values_pred(:,1)), x_axis, transpose(values_naive(:,1)), x_axis, transpose(values_actual(:,2)));
    legend('y = Predicted temperature', 'y = Naive temperature', 'y = Actual temperature','Location','southoutside');
    saveas(gcf, 'temperature2L.fig');

    %plot(1:size(error, 1), transpose(error), 1:size(train_error, 1), transpose(train_error));
    plot(1:size(error, 1), transpose(error));
    legend('y = Squared error', 'Location','southoutside');
    saveas(gcf, 'temp_error.fig');

    save('post_temp_train.mat', 'Winput', 'Wprev1', 'Wprev2', 'Winterior', 'Woutput', 'data', 'std_mean', 'error');

end

