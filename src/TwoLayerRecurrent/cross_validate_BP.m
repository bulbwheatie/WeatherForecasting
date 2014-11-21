% Models the temperature output based on all features

function cross_validate_BP()

    batch_size = 10;
    batches = 1000;
    [data, ~] = getData_struct('all', 4);
    neurons = [4, 8, 16, 32, 64, 128];
    errorn = zeros(length(neurons), 1);
    
    %cross validate on only temperature
    data.trainY = data.trainY(:,2,:);
    data.validateY = data.validateY(:,2,:);
    data.testY = data.testY(:,2,:);
    
    for i=1:length(neurons);
        disp(neurons(i));
        %Random init of weights
        Winput_init = initWeights(size(data.trainX,2), neurons(i),-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
        Winterior_init = initWeights(neurons(i)+1, neurons(i),-1/10, 1/10);
        Wprev1_init = initWeights(neurons(i)+1, neurons(i),-1/10, 1/10);
        Wprev2_init = initWeights(neurons(i)+1, neurons(i),-1/10, 1/10);
        Woutput_init = initWeights(neurons(i)+1, size(data.trainY, 2), -1/2, 1/2); %Only single output feature in this case

        [Winput, Winterior, Wprev1, Wprev2, Woutput, error_vec, validate_vec, test_error] = train_BP(data, Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init, batch_size, batches);
    
        errorn(i,1) = test_error;
    end
    save('cross_validate_BP_neurons.mat', 'errorn', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');

    
    stacks = [4, 6, 8, 10, 12, 14];
    num_neurons = 30;
    errors = zeros(length(stacks), 1);
    
    for i=1:length(stacks);
        disp(stacks(i));
        
        [data, ~] = getData_struct('all', stacks(i));
        %cross validate on only temperature
        data.trainY = data.trainY(:,2,:);
        data.validateY = data.validateY(:,2,:);
        data.testY = data.testY(:,2,:);
        
        %Random init of weights
        Winput_init = initWeights(size(data.trainX,2), num_neurons,-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
        Winterior_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
        Wprev1_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
        Wprev2_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
        Woutput_init = initWeights(num_neurons+1, size(data.trainY, 2), -1/2, 1/2); %Only single output feature in this case

        [Winput, Winterior, Wprev1, Wprev2, Woutput, error_vec, validate_vec, test_error] = train_BP(data, Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init, batch_size, batches);
    
        errors(i,1) = test_error;
        save('cross_validate_BP_stacks.mat', 'errors', 'stacks', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
    end
end

