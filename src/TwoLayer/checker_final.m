% Models the temperature output based on all features

function checker_final()
    batch_size = 10;
    num_stacks = 8;
    num_neurons = 30;
    batches = 1000;
    [data, std_mean] = getData_struct('one', num_stacks);
    num_features = size(data.trainY,2);
    for i=1:num_features
        datas(i) = data;
        datas(i).trainY = data.trainY(:,i,:);
    end
    
    test_error = zeros(size(data.trainX, 1), 1);
        
    %Random init of weights
    Winput_init = initWeights(size(data.trainX,2), num_neurons,-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
    Winterior_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
    Wprev1_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
    Wprev2_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
    Woutput_init = initWeights(num_neurons+1, size(data(1).trainY, 2), -1/2, 1/2); %Only single output feature in this case
    
    Winputs = zeros(size(Winput_init, 1), size(Winput_init, 2), num_features);
    Winteriors = zeros(size(Winterior_init, 1), size(Winterior_init, 2), num_features);
    Wprev1s = zeros(size(Wprev1_init, 1), size(Wprev1_init, 2), num_features);
    Wprev2s = zeros(size(Wprev2_init, 1), size(Wprev2_init, 2), num_features);
    Woutputs = zeros(size(Woutput_init, 1), size(Woutput_init, 2), num_features);
    
    train_errors = zeros(batches, num_features);
    validate_errors = zeros(batches, num_features);

    for i=1:num_features
        disp(i);
        [Winputs(:,:,i), Winteriors(:,:,i), Wprev1s(:,:,i), Wprev2s(:,:,i), Woutputs(:,:,i), train_errors(:, i), validate_errors(:, i), test_error(i)] = train_BP_struct(data(i), Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init, batch_size, batches);
    end
    
    save('final_weights.mat', 'errors', 'stacks', 'error_vec', 'datas', 'Winputs', 'Winteriors', 'Wprev1s', 'Wprev2s', 'Woutputs');
end

