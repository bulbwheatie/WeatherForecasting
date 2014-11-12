% Models the temperature output based on all features

function temp_checker(data, std_mean)
    [Winput, Wprev, Woutput, error] = train(data, 'temp');
    
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

    plot(1:size(error, 1), transpose(error));
    legend('y = Squared error','Location','southoutside');
    saveas(gcf, 'temp_error.fig');

    save('post_temp_train.mat', 'Winput', 'Wprev', 'Woutput', 'data', 'std_mean', 'error');

end

