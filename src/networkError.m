% Calculates the error for the network
% Takes in the input, weight, target output 

% Runs all the data through the system accumulate the error
function netError = networkError(data, W) 
    Ytest = forwardPass(data.input, W);
    netError = 0;
    
    % Accumulate error per sample
    N = (size(Xtest, 2));
    for i = 1:N
        netError = netError +  msError(data.output(i,:), Ytest(i,:))/N;
    end
end