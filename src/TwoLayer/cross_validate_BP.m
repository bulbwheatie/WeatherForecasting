% Models the temperature output based on all features

function cross_validate_BP()
    %[data, std_mean] = getData('one');
    [data, ~] = getData_struct('one', 4);

    batch_size = 10;
%    valid_size = 50;
%    small_set = data(1:1000+num_stacks,2);
%    valid_data = data(2000:2000+valid_size+num_stacks,2);
    
    %Random init of weights
%    small_set = [small_set ones(size(small_set,1), 1)]; % Add bias feature
%    valid_data = [valid_data ones(size(valid_data,1), 1)]; % Add bias feature
    
    neurons = [4, 8, 16, 32, 64, 128];
    errorn = zeros(length(neurons), 1);
    
    for i=1:length(neurons);
        disp(neurons(i));
        %Random init of weights
        Winput_init = initWeights(size(data.trainX,2), neurons(i),-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
        Winterior_init = initWeights(neurons(i)+1, neurons(i),-1/10, 1/10);
        Wprev1_init = initWeights(neurons(i)+1, neurons(i),-1/10, 1/10);
        Wprev2_init = initWeights(neurons(i)+1, neurons(i),-1/10, 1/10);
        Woutput_init = initWeights(neurons(i)+1, size(data.trainY, 2), -1/2, 1/2); %Only single output feature in this case

        [Winput, Winterior, Wprev1, Wprev2, Woutput, train_error, error_vec] = train_BP_struct(data, Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init, batch_size);
    
        errorn(i,1) = error_vec(length(error_vec));
        
        if i==1
            save('cross_validate_BP1n.mat', 'errorn', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==2
            save('cross_validate_BP2n.mat', 'errorn', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==3
            save('cross_validate_BP3n.mat', 'errorn', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==4
            save('cross_validate_BP4n.mat', 'errorn', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==5
            save('cross_validate_BP5n.mat', 'errorn', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==6
            save('cross_validate_BP6n.mat', 'errorn', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        else
            save('cross_validate_BPn.mat', 'errorn', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        end
    end
    
    num_stacks = [2, 4, 6, 8, 10, 12];
    num_neurons = 15;
    errors = zeros(length(neurons), 1);
    
    for i=1:length(num_stacks);
        disp(num_stacks(i));
        
        [data, ~] = getData_struct('one', num_stacks(i));
        %Random init of weights
        Winput_init = initWeights(size(data.trainX,2), num_neurons,-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
        Winterior_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
        Wprev1_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
        Wprev2_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
        Woutput_init = initWeights(num_neurons+1, size(data.trainY, 2), -1/2, 1/2); %Only single output feature in this case

        [Winput, Winterior, Wprev1, Wprev2, Woutput, train_error, error_vec] = train_BP_struct(data, Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init, batch_size);
    
        errors(i,1) = error_vec(length(error_vec));
        
        if i==1
            save('cross_validate_BP1s.mat', 'errors', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==2
            save('cross_validate_BP2s.mat', 'errors', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==3
            save('cross_validate_BP3s.mat', 'errors', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==4
            save('cross_validate_BP4s.mat', 'errors', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==5
            save('cross_validate_BP5s.mat', 'errors', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==6
            save('cross_validate_BP6s.mat', 'errors', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        else
            save('cross_validate_BPs.mat', 'errors', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        end
    end
    
    %error
    plot(neurons, transpose(errorn(:,1)));
    %plot(1:length(train_error_vec), transpose(train_error_vec));
    legend('y = Squared error', 'x = Number of Neurons', 'Location','southeast');
    saveas(gcf, 'cross_val_error_BPn.fig');
        
    %error
    plot(num_stacks, transpose(errors(:,1)));
    %plot(1:length(train_error_vec), transpose(train_error_vec));
    legend('y = Squared error','x = Number of Stacks', 'Location','southeast');
    saveas(gcf, 'cross_val_error_BPs.fig');
end

