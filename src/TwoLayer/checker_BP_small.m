% Models the temperature output based on all features

function checker_BP_small(data, std_mean)
    batch_size = 10;
    num_stacks = 12;
    small_set = data(1:2000+num_stacks,2);

    
    %Random init of weights
    num_neurons  = 25;
    small_set = [small_set ones(size(small_set,1), 1)]; % Add bias feature
    Winput_init = initWeights(2, num_neurons,-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
    Winterior_init = initWeights(num_neurons + 1, num_neurons,-1/10, 1/10);
    Wprev1_init = initWeights(num_neurons + 1, num_neurons,-1/10, 1/10);
    Wprev2_init = initWeights(num_neurons + 1, num_neurons,-1/10, 1/10);
    Woutput_init = initWeights(num_neurons + 1, 1, -1/2, 1/2); %Only single output feature in this case
    
    [Winput, Winterior, Wprev1, Wprev2, Woutput, error] = train_BP(small_set, Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init, 'temp', batch_size, num_stacks);
    
    i = 2;
    X = data(i:i+num_stacks-1,2);
    X_naive = data(i:i+num_stacks-1,2);
    values_pred = zeros(6,1);
    values_naive= zeros(6,1);
    values_actual = data(i+num_stacks:i+num_stacks + 5,2);
    
    for j=1:6
        [temp_y, ~, ~,~, ~] = feedForward([X ones(size(X,1), 1)], Winput, Winterior, Wprev1, Wprev2, Woutput);
        values_pred(j,:) = (temp_y(j,1) .* std_mean(1,2)) + std_mean(2,2);
        values_actual(j,:) = (values_actual(j,1) .* std_mean(1,2)) + std_mean(2,2); %Restore the actual values too
        %values_full = data(i+17+j,:);
        %values_full(1,2) = values_pred(j,:);
        X = [X(2:size(X,1)) ; temp_y(j,1)];

        [naive_y, ~, ~] = feedForward([X_naive ones(size(X_naive,1), 1)], Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init);
        values_naive(j,:) = (naive_y(j,1) .* std_mean(1,2)) + std_mean(2,2);
        %naive_full = data(i+17+j,:);
        %naive_full(1,2) = values_naive(j,:);
        X_naive = [X_naive(2:size(X,1)) ; naive_y(j,1)];
    end
    
    x_axis = 1:size(values_pred, 1);
    
    %temperature -- comapre naive to trained
    plot(x_axis, transpose(values_pred(:,1)), x_axis, transpose(values_naive(:,1)), x_axis, transpose(values_actual(:,1)));
    legend('y = Predicted temperature', 'y = Naive temperature', 'y = Actual temperature','Location','southoutside');
    saveas(gcf, 'graphs/temperature2L.fig');
    
    %error
    plot(1:size(error, 1), transpose(error));
    legend('y = Squared error','Location','southeast');
    saveas(gcf, 'graphs/train_error2L.fig');
    
    save('checker_BP_small.mat', 'Winput', 'Wprev1', 'Wprev2', 'Winterior', 'Woutput', 'data', 'std_mean', 'error');
end

