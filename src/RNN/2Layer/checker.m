
function checker(raw_data, std_mean)
    %std_mean = raw_std_mean(:,2);
    data = [ones(size(raw_data, 1), 1) raw_data(:,2)];
    %Random init of weights
    num_neurons  = 20;
    Winput = initWeights(size(data, 2), num_neurons,-1, 1); % Create a d + 1 x n matrix for the extra bias feature
    Winterior = initWeights(num_neurons, num_neurons,-1, 1);
    Wprev1 = initWeights(num_neurons, num_neurons,-1, 1);
    Wprev2 = initWeights(num_neurons, num_neurons,-1, 1);
    Woutput = initWeights(num_neurons, 1, -2, 2); %Only single output feature in this case
    
    num_stacks = 6;
    epochs = 5;
    
    [Winput, Winterior, Wprev1, Wprev2, Woutput, error] = train_random_new(data, Winput, Winterior, Wprev1, Wprev2, Woutput, num_stacks, epochs);
    
    loaded_data = data;
    save('post_train.mat', 'loaded_data', 'std_mean', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput', 'error', 'num_stacks', 'epochs');

    output_stacks = 3;
    
    for i=[1, 500, 1000, 1500, 2000]
        createGraphs(data, std_mean, Winput, Winterior, Wprev1, Wprev2, Woutput, num_stacks, output_stacks, i);
    end
    
    %error
    plot(1:length(error), transpose(error));
    legend('y = Mean Squared Error','Location','southeast');
    saveas(gcf, 'error.fig');
end