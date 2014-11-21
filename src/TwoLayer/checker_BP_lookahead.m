% Models the temperature output based on all features

function checker_BP_lookahead(data, order_data, std_mean)
    feature_num = 2; %Temperature
    batch_size = 10;
    num_stacks = 8; %data should have num_stacks + lookahead stacks
    lookahead = 1;
    batches = 100;
    data.trainY = data.trainY(:,feature_num, :); %Get a single output feature
    data.validateY = data.validateY(:,feature_num, :);
    data.testY = data.testY(:, feature_num, :);
    
    %Random init of weights
    num_neurons  = 40;
    Winput_init = initWeights(size(data.trainX,2), num_neurons,-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
    Winterior_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
    Wprev1_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
    Wprev2_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
    Woutput_init = initWeights(num_neurons+1, size(data.trainY, 2), -1/2, 1/2); %Only single output feature in this case
   
    [Winput, Winterior, Wprev1, Wprev2, Woutput, train_error, valid_error, test_error] = train_BP_lookahead(data, Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init, batch_size, batches, lookahead, feature_num);
    
    %Do prediction without extra lookahead stacks
    i = 200;
    X = order_data(i:i+num_stacks-1, :); 
    values_pred = zeros(6,1);
    values_actual = order_data(i+num_stacks:i+num_stacks+5, 2); %Predict 6 into future
    
    for j=1:6
        [temp_y, ~, ~,~, ~] = feedForward_lookahead([X ones(size(X,1), 1)], Winput, Winterior, Wprev1, Wprev2, Woutput, 0, feature_num);
        values_pred(j,:) = (temp_y(end,:) .* std_mean(1,2)) + std_mean(2,2);
        values_full = order_data(i+num_stacks + j -1 ,:);
        values_full(1,2) = temp_y(end,:);
        X = [X(2:size(X,1), :) ; values_full];
    end
    
    values_actual = (values_actual .* std_mean(1,2)) + std_mean(2,2); %Restore the actual values too
    
    x_axis = 1:size(values_pred, 1);
    
    %temperature -- comapre naive to trained
    plot(x_axis, transpose(values_pred(:,1)), x_axis, transpose(values_actual(:,1)));
    legend('y = Predicted temperature', 'y = Actual temperature','Location','southoutside');
    xlabel('Time (hours)');
    ylabel('Temperature');
    title('Predicted Tempearture on Small Set');
    saveas(gcf, 'graphs/temperature2L.fig');
    
    %error
    plot(1:size(train_error, 1), transpose(train_error), 1:size(valid_error,1), transpose(valid_error));
    legend('y = Squared train error', 'Y = Squared valid error', 'Location','southeast');
    xlabel('Iterations');
    ylabel('Squared Error');
    title('Training and Validation Error for Batch Train');
    saveas(gcf, 'graphs/train_error2L.fig');

    save('checker_BP_lookahead.mat', 'Winput', 'Wprev1', 'Wprev2', 'Winterior', 'Woutput', 'data', 'std_mean', 'train_error', 'valid_error');
end

