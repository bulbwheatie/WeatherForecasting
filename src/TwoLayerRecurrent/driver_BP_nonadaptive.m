% Trains a network for every output feature using backpropagation with a
% fixed learning rate.  Saves the weight matrices and error vectors from
% every network into BP.mat

%INPUT
% mode = 'one' or 'all' and determines whether the networks are trained on the 
% data from just one city or the data from all the cities

%SAVED VARIABLES
% test_errors = a 6 x 1 vector where test_errors(i,1) contains the test error for the matrix trained for feature i
% train_errors = a batches x 6 matrix where train_errors(:,i) is the train_error vector returned by train_BP
% validate_errors = a batches x 6 matrix where validate_errors(:,i) is the valid_error vector returned by train_BP
% datas = an array of the 6 data structs where datas(i) = data for feature i
% Winputs = a 3d matrix where Winputs(:,:,i) is the Winput weight matrix for feature i
% Winteriors = a 3d matrix where Winterior(:,:,i) is the Winterior weight matrix for feature i
% Wprev1s = a 3d matrix where Wprev1(:,:,i) is the Wprev1 weight matrix for feature i
% Wprev2s = a 3d matrix where Wprev2(:,:,i) is the Wprev2 weight matrix for feature i
% Woutputs = a 3d matrix where Woutputs(:,:,i) is the Woutput weight matrix for feature i
% std_mean = a 2x6 matrix where std_mean(1,i) is the standard deviation of feature i and std_mean(2,i) is the mean of feature i

function driver_BP_nonadaptive(mode)
    batch_size = 100; %number of samples to use per batch in the train file
    num_stacks = 8; %number of stacks to use for the network
    num_neurons = 30; %number of neurons to use in each hidden layer
    batches = 5000; %maximum number of iterations to run
    [data, std_mean] = getData_struct(mode, num_stacks);
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

    %initialize 3d weight matrices that will have the weight matrix as the first two
    %dimensions and the feature that the matrix is for as the third dimension
    Winputs = zeros(size(Winput_init, 1), size(Winput_init, 2), num_features);
    Winteriors = zeros(size(Winterior_init, 1), size(Winterior_init, 2), num_features);
    Wprev1s = zeros(size(Wprev1_init, 1), size(Wprev1_init, 2), num_features);
    Wprev2s = zeros(size(Wprev2_init, 1), size(Wprev2_init, 2), num_features);
    Woutputs = zeros(size(Woutput_init, 1), size(Woutput_init, 2), num_features);
    
    train_errors = zeros(floor(batches*batch_size/(batch_size-1))+1, num_features);
    validate_errors = zeros(floor(batches*batch_size/(batch_size-1))+1, num_features);
    test_errors = zeros(size(data.trainX, 2), 1);

    %train a network for every feature
    for i=1:num_features
        disp(i);
        [Winput, Winterior, Wprev1, Wprev2, Woutput, train_error, validate_error, test_error] = train_BP_nonadaptive(datas(i), Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init, batch_size, batches);
        Winputs(:,:,i) = Winput;
        Winteriors(:,:,i) = Winterior;
        Wprev1s(:,:,i) = Wprev1;
        Wprev2s(:,:,i) = Wprev2;
        Woutputs(:,:,i) = Woutput;
        train_errors(:, i) = train_error;
        validate_errors(:, i) = validate_error;
        test_errors(i) = test_error;
    end
    
    %save all the data
    save('BP_nonadaptive.mat', 'test_errors', 'train_errors', 'validate_errors', 'datas', 'Winputs', 'Winteriors', 'Wprev1s', 'Wprev2s', 'Woutputs', 'std_mean');
end

