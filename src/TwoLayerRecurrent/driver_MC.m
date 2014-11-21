% Trains a network for every output feature using the Monte Carlo update
% method and saves the weight matrices and error vectors from every network into BP.mat

%INPUT
% mode = 'one' or 'all' and determines whether the networks are trained on the 
% data from just one city or the data from all the cities

%SAVED VARIABLES
% errors = an epochs x 6 matrix where errors(:,i) is the error vector returned by train_MC
% datas = an array of the 6 data structs where datas(i) = data for feature i
% Winputssa = a 3d matrix where Winputs(:,:,i) is the Winput weight matrix for feature i
% Winteriorsa = a 3d matrix where Winterior(:,:,i) is the Winterior weight matrix for feature i
% Wprev1sa = a 3d matrix where Wprev1(:,:,i) is the Wprev1 weight matrix for feature i
% Wprev2sa = a 3d matrix where Wprev2(:,:,i) is the Wprev2 weight matrix for feature i
% Woutputsa = a 3d matrix where Woutputs(:,:,i) is the Woutput weight matrix for feature i
% std_mean = a 2x6 matrix where std_mean(1,i) is the standard deviation of feature i and std_mean(2,i) is the mean of feature i

function driver_MC(mode)
    num_stacks = 8; %number of stacks to use for the network
    num_neurons = 10; %number of neurons to use in each hidden layer
    epochs = 15; %maximum number of epochs to run
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
    Winputsa = zeros(size(Winput_init, 1), size(Winput_init, 2), num_features);
    Winteriorsa = zeros(size(Winterior_init, 1), size(Winterior_init, 2), num_features);
    Wprev1sa = zeros(size(Wprev1_init, 1), size(Wprev1_init, 2), num_features);
    Wprev2sa = zeros(size(Wprev2_init, 1), size(Wprev2_init, 2), num_features);
    Woutputsa = zeros(size(Woutput_init, 1), size(Woutput_init, 2), num_features);
    
    errors = zeros(epochs, num_features);

    %train a network for every feature
    for i=1:num_features
        disp(i);
        [Winput, Winterior, Wprev1, Wprev2, Woutput, error] = train_MC(datas(i), Winput_init, Winterior_init, Wprev1_init, Wprev2_init, Woutput_init, epochs);
        Winputsa(:,:,i) = Winput;
        Winteriorsa(:,:,i) = Winterior;
        Wprev1sa(:,:,i) = Wprev1;
        Wprev2sa(:,:,i) = Wprev2;
        Woutputsa(:,:,i) = Woutput;
        errors(:, i) = error;
    end
    
    %save the values
    save('MC.mat', 'errors', 'datas', 'Winputsa', 'Winteriorsa', 'Wprev1sa', 'Wprev2sa', 'Woutputsa', 'std_mean');
end