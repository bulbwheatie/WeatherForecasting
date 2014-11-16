
function checker(data, std_mean) 
    batch_size = 12;
    num_stacks = 12;
    
    %Random init of weights
    num_neurons  = 25;
    X = [ones(size(X,1), 1) X]; % Add bias feature
    Winput = initWeights(size(X, 2), num_neurons,-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
    Winterior = initWeights(num_neurons, num_neurons,-1/10, 1/10);
    Wprev1 = initWeights(num_neurons, num_neurons,-1/10, 1/10);
    Wprev2 = initWeights(num_neurons, num_neurons,-1/10, 1/10);
    Woutput = initWeights(num_neurons, size(Y,2), -1/2, 1/2);
    
    [Winput, Winterior, Wprev1, Wprev2, Woutput, error] = train_new(X, Winput, Winterior, Wprev1, Wprev2, Woutput, 'temp', batch_size, num_stacks);
    
    i = 1;
    X = data(i:i+11,:);
    values_pred = zeros(6,6);
    values_actual = data(i+12:i+17,:);
    
    for j=1:6
        [temp_y, signals] = feedForward_new([ones(size(X,1), 1) X], Winput, Wprev, Woutput);
        values_pred(j,:) = (temp_y(size(temp_y,1),:) .* std_mean(1,:)) + std_mean(2,:);
        values_actual(j,:) = (values_actual(j,:) .* std_mean(1,:)) + std_mean(2,:); %Restore the actual values too
        X = [X(2:size(X,1),:) ; values_pred(j,:)];
    end
    
    x_axis = 1:size(values_pred, 1);
    
    %temperature
    plot(x_axis, transpose(values_pred(:,2)), x_axis, transpose(values_actual(:,2)));
    legend('y = Predicted temperature','y = Actual temperature','Location','southeast');
    saveas(gcf, 'temperature.fig');
    
    %dewpoint
    plot(x_axis, transpose(values_pred(:,3)), x_axis, transpose(values_actual(:,3)));
    legend('y = Predicted dewpoint','y = Actual dewpoint','Location','southeast');
    saveas(gcf, 'dewpoint.fig');
    
    %windspeed
    plot(x_axis, transpose(values_pred(:,4)), x_axis, transpose(values_actual(:,4)));
    legend('y = Predicted windspeed','y = Actual windspeed','Location','southeast');
    saveas(gcf, 'windspeed.fig');
    
    %pressure
    plot(x_axis, transpose(values_pred(:,6)), x_axis, transpose(values_actual(:,6)));
    legend('y = Predicted pressure','y = Actual pressure','Location','southeast');
    saveas(gcf, 'pressure.fig');
    
    %error
    plot(1:size(error, 1), transpose(error));
    legend('y = Squared error','Location','southeast');
    saveas(gcf, 'error.fig');
    
    save('post_train.mat', 'Winput', 'Wprev', 'Woutput', 'data', 'std_mean', 'error');
end