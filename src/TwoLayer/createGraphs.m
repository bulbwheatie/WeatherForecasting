function [values_pred, values_actual] = createGraphs(data, std_mean, Winput, Winterior, Wprev1, Wprev2, Woutput, num_stacks, output_stacks, row, col, name)
    X = data(row:row+num_stacks-1,:);

    values_pred = zeros(output_stacks,size(Woutput,2));
    values_actual = data(row+num_stacks:row+num_stacks+output_stacks-1, 2:size(X,2));

    for j=1:output_stacks
        [temp_y, ~, ~] = feedForward_new(X, Winput, Winterior, Wprev1, Wprev2, Woutput);
        values_pred(j,:) = (temp_y(size(temp_y,1),:) .* std_mean(1,:)) + std_mean(2,:);
        values_actual(j,:) = (values_actual(j,:) .* std_mean(1,:)) + std_mean(2,:); %Restore the actual values too
        X = [X(2:size(X,1),:) ; 1 temp_y(size(temp_y,1),:)];
    end
    disp(values_pred);
    disp(values_actual);
    x_axis = (1:size(values_pred, 1))*4;

    %temperature
    plot(x_axis, transpose(values_pred(:,col)), x_axis, transpose(values_actual(:,col)));
    legend(strcat('y = Predicted ', name),strcat('y = Actual ', name), 'Location', 'southeast');
    saveas(gcf, strcat(name, num2str(row), '.fig'), 'fig');
end

