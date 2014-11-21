% createGraphs.m
% 
% Takes in weight matrices and predicts data by feeding through network
%
% Input: 
% Winput = [m x n x s] where s = number of stacks, n = number of features, m = number of samples
% data = [s x n] data to predict on (zero-meaned)
% std_mean = [2 x n] where n = number of features, 1st row = std, 2nd row = mean
% Y_actual= the next 24 hours of data in 4 hour intervals
% data = the initial data set to predict on 

function [Ypred, Yactual, test_error] = createGraphs_combination(index, Winput, Winterior, Wprev1, Wprev2, Woutput, Winputa, Winteriora, Wprev1a, Wprev2a, Woutputa)
    [data_full, std_mean] = getData('test');
    data_init = data_full(index:index+7, :);
    output_stacks = 6;
    Yactual = data_full(index+8:index+7 + output_stacks,:);
    data = [data_init ones(size(data_init,1),1)];
    num_features = size(Winput, 1)-1;
    Ypred = zeros(output_stacks, num_features); %Predict 6 data points into the future
    new_data_row = ones(1, num_features);
    %For each feature, predict for the current time iteration
    %take all the predicted features to make the new data point 
    for i=1:output_stacks
        %For each feature use it's weight matrix to predict the next value
        for j=1:num_features
            [temp_y, ~, ~] = feedForward(data, Winput(:,:,j), Winterior(:,:,j), Wprev1(:,:,j), Wprev2(:,:,j), Woutput(:,:,j));
            [temp_ya, ~, ~] = feedForward(data, Winputa(:,:,j), Winteriora(:,:,j), Wprev1a(:,:,j), Wprev2a(:,:,j), Woutputa(:,:,j));
            temp_y = (temp_ya + temp_y)/2;
            new_data_row(1,j) = temp_y(end,1);
        end
        Ypred(i,:) = (new_data_row .* std_mean(1,:)) + std_mean(2,:); %convert back to the original data
        data = [data(2:end,:) ; new_data_row 1]; %Create the new input data series
    end

    %Calculate the actual values
    for i=1:output_stacks
        Yactual(i,:) = (Yactual(i,:) .* std_mean(1,:)) + std_mean(2,:);
    end
    
    test_error = (Yactual - Ypred).^2; %Calculate the test error for each feature
    disp(Ypred);
    disp(Yactual);
    disp(test_error);
    
    for i=1:size(data_init,1)
        data_init(i,:) = (data_init(i,:) .* std_mean(1,:)) + std_mean(2,:);
    end
    
    x_axis_pred = [0 (1:size(Ypred, 1))*4];
    x_axis_actual = ((1:size(Ypred, 1)+size(data_init,1))-size(data_init,1))*4;
    axes = ones(num_features, 4);
    axes(1,:) = [-28 24 0 12];
    axes(2,:) = [-28 24 -20 30];
    axes(3,:) = [-28 24 -10 20];
    axes(4,:) = [-28 24 0 20];
    axes(5,:) = [-28 24 0 360];
    axes(6,:) = [-28 24 28.5 30];
    features = {'Visibility', 'Temperature (C)', 'Dew Point', 'Wind Speed (mph)', 'Wind Direction', 'Pressure'};
    %Plot each feature's predicted versus actual value
    for i = 1:num_features
    	plot(x_axis_pred, [data_init(end,i) Ypred(:,i)'], x_axis_actual, [data_init(:,i)' Yactual(:,i)']);
    	legend(strcat('y = Predicted ', features{i}), strcat('y = Actual ', features{i}), 'Location', 'southeast');
        xlabel('Time');
        ylabel(features{i});
        title(strcat('Predicted vs. Actual ', features{i}));
        axis(axes(i,:));
    	saveas(gcf, strcat('graphs/', features{i}, '_allData.fig'), 'fig');
    end
end

