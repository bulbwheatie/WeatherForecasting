function checker_MC()
    num_stacks = 6;
    num_neurons = 10;
    epochs = 15;
    [data, std_mean] = getData_struct('one', num_stacks);
    num_features = size(data.trainY,2);
    for i=1:num_features
        datas(i) = data;
        datas(i).trainY = data.trainY(:,i,:);
        datas(i).validateY = data.validateY(:,i,:);
        datas(i).testY = data.testY(:,i,:);
    end
    
    %Random init of weights
    Winput_init = initWeights(size(data.trainX,2), num_neurons,-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
    Winterior_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
    Wprev1_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
    Wprev2_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
    Woutput_init = initWeights(num_neurons+1, size(datas(1).trainY, 2), -1/2, 1/2); %Only single output feature in this case

    Winputs = zeros(size(Winput_init, 1), size(Winput_init, 2), num_features);
    Winteriors = zeros(size(Winterior_init, 1), size(Winterior_init, 2), num_features);
    Wprev1s = zeros(size(Wprev1_init, 1), size(Wprev1_init, 2), num_features);
    Wprev2s = zeros(size(Wprev2_init, 1), size(Wprev2_init, 2), num_features);
    Woutputs = zeros(size(Woutput_init, 1), size(Woutput_init, 2), num_features);
    
    errors = zeros(epochs, num_features);

    for i=1:num_features
        disp(i);
        [Winput, Winterior, Wprev1, Wprev2, Woutput, error] = train_MC(datas(i), Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init, epochs);
        Winputs(:,:,i) = Winput;
        Winteriors(:,:,i) = Winterior;
        Wprev1s(:,:,i) = Wprev1;
        Wprev2s(:,:,i) = Wprev2;
        Woutputs(:,:,i) = Woutput;
        errors(:, i) = error;
    end
    
    save('MC.mat', 'errors', 'datas', 'Winputs', 'Winteriors', 'Wprev1s', 'Wprev2s', 'Woutputs', 'std_mean');
end