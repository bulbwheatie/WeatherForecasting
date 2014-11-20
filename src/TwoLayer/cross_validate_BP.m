% Models the temperature output based on all features

function cross_validate_BP()
    [data, ~] = getData_struct('one');
    
    %neurons = [2, 4, 8, 16, 32, 64];
    neurons = [5, 10, 50];
    error = zeros(length(neurons), 1);
    max_epochs = 500;
    
    for i=1:length(neurons);
        %Random init of weights
        Winput_init = initWeights(size(data.trainX,2), neurons(i),-1/10, 1/10); % Create a d + 1 x n matrix for the extra bias feature
        Winterior_init = initWeights(neurons(i)+1, neurons(i),-1/10, 1/10);
        Wprev1_init = initWeights(neurons(i)+1, neurons(i),-1/10, 1/10);
        Wprev2_init = initWeights(neurons(i)+1, neurons(i),-1/10, 1/10);
        Woutput_init = initWeights(neurons(i)+1, size(data.trainY, 2), -1/2, 1/2); %Only single output feature in this case

        [Winput, Winterior, Wprev1, Wprev2, Woutput, error_vec] = train_BP_adaptive(data, Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init, max_epochs);

        error(i,1) = error_vec(length(error_vec));
        
        if i==1
            save('cross_validate_BP1.mat', 'error', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==2
            save('cross_validate_BP2.mat', 'error', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==3
            save('cross_validate_BP3.mat', 'error', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==4
            save('cross_validate_BP4.mat', 'error', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==5
            save('cross_validate_BP5.mat', 'error', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        elseif i==6
            save('cross_validate_BP6.mat', 'error', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        else
            save('cross_validate_BP6.mat', 'error', 'neurons', 'error_vec', 'data', 'Winput', 'Winterior', 'Wprev1', 'Wprev2', 'Woutput');
        end
    end
    
    %error
    plot(neurons, transpose(error(:,1)));
    %plot(1:length(train_error_vec), transpose(train_error_vec));
    legend('y = Squared error','Location','southeast');
    saveas(gcf, 'cross_val_error_BP.fig');
end

