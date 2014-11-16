
function checker(data, std_mean) 
    [Winput, Wprev, Woutput, error] = train(data, 'all', 1);
    i = 1;
    X = data(i:i+11,:);
    values_pred = zeros(6,6);
    values_actual = data(i+12:i+17,:);
    
    for j=1:6
        [temp_y, signals] = feedForward([ones(size(X,1), 1) X], Winput, Wprev, Woutput);
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