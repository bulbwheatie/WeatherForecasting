function train_error = trainingErrorChecker(data, std_mean)
    % Create a small data set and run the entire set through the training
    % process for each update and check that the error is indeed reducing.
    batch_size = 100;
    small_set = data(1:batch_size,:);
    [Winput, Wprev, Woutput, error, train_error] = train(small_set, 'temp', batch_size);
    
    i = 1;
    X = data(i:i+11,:);
    values_pred = zeros(6,1);
    values_actual = data(i+12:i+17,:);
    
    for j=1:6
        [temp_y, signals] = feedForward([ones(size(X,1), 1) X], Winput, Wprev, Woutput);
        values_pred(j,:) = (temp_y(size(temp_y,1),:) .* std_mean(1,2)) + std_mean(2,2);
        values_actual(j,:) = (values_actual(j,:) .* std_mean(1,:)) + std_mean(2,:); %Restore the actual values too
        values_full = data(i+17+j,:);
        values_full(1,2) = values_pred(j,:);
        X = [X(2:size(X,1),:) ; values_full];
    end
    
    x_axis = 1:size(values_pred, 1);
    
    %temperature
    plot(x_axis, transpose(values_pred(:,1)), x_axis, transpose(values_actual(:,2)));
    legend('y = Predicted temperature','y = Actual temperature','Location','southoutside');
    saveas(gcf, 'temperature.fig');
    
    %error
    plot(1:size(train_error, 1), transpose(train_error));
    legend('y = Squared error','Location','southeast');
    saveas(gcf, 'train_error.fig');
end