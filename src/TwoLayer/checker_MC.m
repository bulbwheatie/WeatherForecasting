
function checker_MC(raw_data, std_mean)
    %std_mean = raw_std_mean(:,2);
    data = [ones(size(raw_data, 1), 1) raw_data];
    for num_neurons=[2, 5, 10, 20, 40, 80, 160]
        %Random init of weights
        Winput = initWeights(size(data, 2), num_neurons,-1, 1); % Create a d + 1 x n matrix for the extra bias feature
        Winterior = initWeights(num_neurons, num_neurons,-1, 1);
        Wprev1 = initWeights(num_neurons, num_neurons,-1, 1);
        Wprev2 = initWeights(num_neurons, num_neurons,-1, 1);
        Woutput = initWeights(num_neurons, size(raw_data, 2), -2, 2);

        num_stacks = 6;
        epochs = 8;

        [Winput, Winterior, Wprev1, Wprev2, Woutput, error] = train_MC(data, Winput, Winterior, Wprev1, Wprev2, Woutput, num_stacks, epochs);
        
        loaded_data = data;
        save(strcat(str2double(num_neurons), 'post_checker_MC.mat'), 'loaded_data', 'std_mean', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput', 'error', 'num_stacks', 'epochs');


        %error
        plot(1:length(error), transpose(error));
        legend('y = Mean Squared Error','Location','southeast');
        saveas(gcf, strcat(str2double(num_neurons), 'error.fig'));
    end
    
     %output_stacks = 3;

    
    %for i=[1, 500, 1000, 1500, 2000]
    %    createGraphs(data, std_mean, Winput, Winterior, Wprev1, Wprev2, Woutput, num_stacks, output_stacks, i, 1, 'Visibility');
    %    createGraphs(data, std_mean, Winput, Winterior, Wprev1, Wprev2, Woutput, num_stacks, output_stacks, i, 2, 'Temperature');
    %    createGraphs(data, std_mean, Winput, Winterior, Wprev1, Wprev2, Woutput, num_stacks, output_stacks, i, 3, 'Dewpoint');
    %    createGraphs(data, std_mean, Winput, Winterior, Wprev1, Wprev2, Woutput, num_stacks, output_stacks, i, 4, 'Windspeed');
    %    createGraphs(data, std_mean, Winput, Winterior, Wprev1, Wprev2, Woutput, num_stacks, output_stacks, i, 5, 'WindDirection');
    %    createGraphs(data, std_mean, Winput, Winterior, Wprev1, Wprev2, Woutput, num_stacks, output_stacks, i, 6, 'Pressure');
    %end
end