% Models the temperature output based on all features

function cross_validate_BP()

      batch_size = 10;
%     [data, ~] = getData_struct('one', 4);
% 

%     
%     neurons = [4, 8, 16, 32, 64, 128];
%     errorn = zeros(length(neurons), 1);
%     
%     for i=1:length(neurons);
%         disp(neurons(i));
%         %Random init of weights
%         Winput_init = initWeights(size(data.trainX,2), neurons(i),-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
%         Winterior_init = initWeights(neurons(i)+1, neurons(i),-1/10, 1/10);
%         Wprev1_init = initWeights(neurons(i)+1, neurons(i),-1/10, 1/10);
%         Wprev2_init = initWeights(neurons(i)+1, neurons(i),-1/10, 1/10);
%         Woutput_init = initWeights(neurons(i)+1, size(data.trainY, 2), -1/2, 1/2); %Only single output feature in this case
% 
%         [Winput, Winterior, Wprev1, Wprev2, Woutput, error_vec, validate_vec, test_error] = train_BP_struct(data, Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init, batch_size);
%     
%         errorn(i,1) = test_error;
%         
%         if i==1
%             save('cross_validate_BP1n.mat', 'errorn', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
%         elseif i==2
%             save('cross_validate_BP2n.mat', 'errorn', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
%         elseif i==3
%             save('cross_validate_BP3n.mat', 'errorn', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
%         elseif i==4
%             save('cross_validate_BP4n.mat', 'errorn', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
%         elseif i==5
%             save('cross_validate_BP5n.mat', 'errorn', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
%         elseif i==6
%             save('cross_validate_BP6n.mat', 'errorn', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
%         else
%             save('cross_validate_BPn.mat', 'errorn', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
%         end
%     end
    
    stacks = [4, 6, 8, 10, 12, 14];
    num_neurons = 30;
    errors = zeros(length(stacks), 1);
    
    for i=1:length(stacks);
        disp(stacks(i));
        
        [data, ~] = getData_struct('one', stacks(i));
        %Random init of weights
        Winput_init = initWeights(size(data.trainX,2), num_neurons,-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
        Winterior_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
        Wprev1_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
        Wprev2_init = initWeights(num_neurons+1, num_neurons,-1/10, 1/10);
        Woutput_init = initWeights(num_neurons+1, size(data.trainY, 2), -1/2, 1/2); %Only single output feature in this case

        [Winput, Winterior, Wprev1, Wprev2, Woutput, error_vec, validate_vec, test_error] = train_BP_struct(data, Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init, batch_size);
    
        errors(i,1) = test_error;
        
        if i==1
            save('cross_validate_BP1s.mat', 'errors', 'stacks', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==2
            save('cross_validate_BP2s.mat', 'errors', 'stacks', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==3
            save('cross_validate_BP3s.mat', 'errors', 'stacks', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==4
            save('cross_validate_BP4s.mat', 'errors', 'stacks', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==5
            save('cross_validate_BP5s.mat', 'errors', 'stacks', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==6
            save('cross_validate_BP6s.mat', 'errors', 'stacks', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        else
            save('cross_validate_BPs.mat', 'errors', 'stacks', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        end
    end
end

