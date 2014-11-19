function crossval_neurons_MC()
    [raw_data, ~] = getData('one');
    data = [ones(size(raw_data, 1), 1) raw_data];
    neurons = [4, 8, 16, 32, 64];
    error = zeros(length(neurons), 1);
    num_stacks = 4;
    epochs = 1;
    for i=1:length(neurons)
        %Random init of weights
        Winput = initWeights(size(data, 2), neurons(i),-1, 1); % Create a d + 1 x n matrix for the extra bias feature
        Winterior = initWeights(neurons(i), neurons(i),-1, 1);
        Wprev1 = initWeights(neurons(i), neurons(i),-1, 1);
        Wprev2 = initWeights(neurons(i), neurons(i),-1, 1);
        Woutput = initWeights(neurons(i), size(raw_data, 2), -2, 2);

        [Winput, Winterior, Wprev1, Wprev2, Woutput, error_vec] = train_MC(data, Winput, Winterior, Wprev1, Wprev2, Woutput, num_stacks, epochs);

        %error
        error(i) = error_vec(length(error_vec));
    end
    save('crossval_neuron_error.mat', 'error', 'neurons');
    plot(neurons, transpose(error));
end
