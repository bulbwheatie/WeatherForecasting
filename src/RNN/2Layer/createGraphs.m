function [values_pred, values_actual] = createGraphs(data, std_mean, Winput, Winterior, Wprev1, Wprev2, Woutput, num_stacks, output_stacks, i)
    X = data(i:i+num_stacks-1,:);

    values_pred = zeros(output_stacks,1);
    values_actual = data(i+num_stacks:i+num_stacks+output_stacks-1, 2);

    for j=1:output_stacks
        [temp_y, ~, ~] = feedForward_new(X, Winput, Winterior, Wprev1, Wprev2, Woutput);
        values_pred(j,1) = (temp_y(size(temp_y,1),1) .* std_mean(1,2)) + std_mean(2,2);
        values_actual(j,1) = (values_actual(j,1) .* std_mean(1,2)) + std_mean(2,2); %Restore the actual values too
        X = [X(2:size(X,1),1:2) ; 1 values_pred(j,1)];
    end
    disp(values_pred);
    disp(values_actual);
    x_axis = (1:size(values_pred, 1))*4;

    %temperature
    plot(x_axis, transpose(values_pred(:,1)), x_axis, transpose(values_actual(:,1)));
    legend('y = Predicted temperature','y = Actual temperature','Location','southeast');
    saveas(gcf, strcat('temperature', num2str(i), '.fig'), 'fig');
end

