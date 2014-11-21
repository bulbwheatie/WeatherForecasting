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

function [values_pred, values_actual, test_error] = createGraphs(data, Yactual, std_mean, Winput, Winterior, Wprev1, Wprev2, Woutput)
    [m x n x s] = size(Winput); % Winput(:,:,j) will give you the weight matrix for the jth feature
    output_stacks = 6;
    Ypred = zeros(output_stacks, n); %Predict 6 data points into the future
    curr_pred = ones(1,n + 1); %current prediction step (plus a bias feature)
    
    %For each feature, predict for the current time iteration
    %take all the predicted features to make the new data point 
    for i=1:output_stacks
        %For each feature use it's weight matrix to predict the next value
        for j=1:n
            [temp_y, ~, ~] = feedForward(data, Winput(:,:,j), Winterior(:,:,j), Wprev1(:,:,j), Wprev2(:,:,j), Woutput(:,:,j));
            curr_pred(1,j) = (temp_y(end,:) .* std_mean(1,j)) + std_mean(2,j); %Normalize to the jth feature in the original set
        end
        data = [data(2:end,:) ; 1 curr_pred]; %Create the new input data series
    end

    %Calculate the actual values
    for i=1:output_stacks
	Yactual(i,:) = Yactual(i,:) * std_mean(1,:) + std_mean(2,:);
    end


    test_error = (Yactual - Ypred).^2; %Calculate the test error for each feature
    disp(Ypred);
    disp(Yactual);
    disp(test_error);

    x_axis = (1:size(values_pred, 1));

    % --PILLS!!!! Are these in the right order??
    features = ['Dew Point', 'Temperature (C)', 'Humidity','Wind Speed (mph)','Wind Direction','Pressure']
    
    %Plot each feature's predicted versus actual value
    for i = 1:n
    	plot(x_axis, transpose(values_pred(:,i)), x_axis, transpose(values_actual(:,i)));
    	legend(strcat('y = Predicted ', features(i),strcat('y = Actual ', features(i)), 'Location', 'southeast');
        xlabel('Time');
	ylabel(features(i));
	title(strcat('Predicted vs. Actual ', features(i));
    	saveas(gcf, strcat('graphs/', features(i), '_allData.fig'), 'fig');
    end
end

