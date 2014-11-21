
function driver_BP()
    [raw_data, raw_std_mean] = getData;
    std_mean = raw_std_mean(:,2);
    data = [ones(size(raw_data, 1), 1) raw_data(:,2)];
    %Random init of weights
    num_neurons  = 25;
    Winput = initWeights(size(data, 2), num_neurons,-1, 1); % Create a d + 1 x n matrix for the extra bias feature
    Wprev = initWeights(num_neurons, num_neurons,-1, 1);
    Woutput = initWeights(num_neurons, 1, -2, 2); %Only single output feature in this case
    
    num_stacks = 8;
    epochs = 100;
    
    %train_random uses a monte carlo update system
    [Winput, Wprev, Woutput, error] = train_BP(data, Winput, Wprev, Woutput, 'temp', num_stacks, epochs);
    
    output_stacks = 2;
    
    for i=[1 500 1000 1500 2000]
        X = data(i:i+num_stacks-1,:);

        values_pred = zeros(output_stacks,1);
        values_actual = data(i+num_stacks:i+num_stacks+output_stacks-1, 2);

        for j=1:output_stacks
            [temp_y, signals] = feedForward(X, Winput, Wprev, Woutput);
            values_pred(j,1) = (temp_y(size(temp_y,1),1) .* std_mean(1,1)) + std_mean(2,1);
            values_actual(j,1) = (values_actual(j,1) .* std_mean(1,1)) + std_mean(2,1); %Restore the actual values too
            X = [X(2:size(X,1),:) ; 1 values_pred(j,1)];
        end
        disp(values_pred);
        disp(values_actual);
        x_axis = (1:size(values_pred, 1))*4;

        %temperature
        plot(x_axis, transpose(values_pred(:,1)), x_axis, transpose(values_actual(:,1)));
        legend('y = Predicted temperature','y = Actual temperature','Location','southeast');
        name = strcat('temperature', num2str(i), '.fig');
        saveas(gcf, name);
    end
    
    %error
    plot(1:length(error), transpose(error));
    legend('y = Mean Squared Error','Location','southeast');
    saveas(gcf, 'error.fig');
    
    save('post_train.mat', 'Winput', 'Wprev', 'Woutput', 'error');
end