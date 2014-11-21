function error = checker_BP(data, std_mean)
    % Create a small data set and run the entire set through the training
    % process for each update and check that the error is indeed reducing.
    feature_num = 2; %Temperature
    batch_size = 10;
    num_stacks = 8;
    batches = 50;
    data.trainY = data.trainY(:,2, :);
    data.validateY = data.validateY(:,2, :);
    data.testY = data.testY(:, 2, :);

    %Random init of weights
    num_neurons  = 30;
    Winput_init = initWeights(size(data.trainX,2), num_neurons,-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
    Winterior_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
    Wprev1_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
    Wprev2_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
    Woutput_init = initWeights(num_neurons+1, size(data.trainY, 2), -1/2, 1/2); %Only single output feature in this case
   
    [Winput, Winterior, Wprev1, Wprev2, Woutput, train_error, valid_error, test_error] = train_BP_adaptive(data, Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init, batch_size, batches);
    
    i = 1;
    X = data.testX(:, :, i:i+num_stacks-1);
    values_pred = zeros(6,1);
    values_actual = data.testY(:, :, i:i+num_stacks-1);
    
    for j=1:6
        [temp_y, ~, ~,~, ~] = feedForward(X, Winput, Winterior, Wprev1, Wprev2, Woutput);
        values_pred(j,:) = (temp_y(j,1) .* std_mean(1,2)) + std_mean(2,2);
        values_actual(j,:) = (values_actual(j,1) .* std_mean(1,2)) + std_mean(2,2); %Restore the actual values too
        values_full = data.trainX(:,:,i+num_stacks-1+j);
        values_full(1,feature_num) = values_pred(j,:); 
        X = [X(2:size(X,1)) ; values_full];
    end
    
    x_axis = (1:size(values_pred, 1) )* 4;
    
    %temperature -- comapre naive to trained
    plot(x_axis, transpose(values_pred(:,1)), x_axis, transpose(values_actual(:,1)));
    legend('y = Predicted temperature', 'y = Naive temperature', 'y = Actual temperature','Location','southoutside');
    xlabel('Time (hours)');
    ylabel('Temperature(C)');
    saveas(gcf, 'graphs/temperature2L.fig');
    
    %error
    plot(1:size(train_error, 1), transpose(train_error), 1:size(valid_error,1), transpose(valid_error));
    legend('y = Squared train error', 'Y = Squared valid error', 'Location','southeast');
    xlabel('Iterations');
    ylabel('Squared Error');
    title('Training and Validation Error for Batch Train');
    saveas(gcf, 'graphs/train_error2L.fig');
    
    save('checker_BP.mat', 'Winput', 'Wprev1', 'Wprev2', 'Winterior', 'Woutput', 'data', 'std_mean', 'train_error', 'valid_error', 'test_error');
    
end