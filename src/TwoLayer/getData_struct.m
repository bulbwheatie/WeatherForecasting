%Takes either 'all' or 'one' as a mode, which gets either all the data or
%just one city's worth.
%returns the matrix containing the standard deviation and the mean of the
%data, and a data a struct that has:
%trainX: a 3-d matrix of the input values, with the stack as the first value,
%   the features, including bias at the end, as the second value,
%   and the row as the third value.
%trainY: a 3-d matrix of the output values, with the stack as the first value,
%   the features as the second value,
%   and the row as the third value.
%validateX, validateY, trainX, and trainY are all in the same format.

function [data, std_mean] = getData_struct(mode, num_stacks)
    if strcmp(mode, 'all')
        files = dir('../../data/*.txt');
    else
        files = dir('../../data/*WA.txt');
    end
    
    num_features = 6;
    num_stacks = 4;
    samples_per_file = 365 * 24/4 - (2 * num_stacks) + 1;
    
    dataX = zeros(num_stacks, num_features+1, samples_per_file);
    dataY = zeros(num_stacks, num_features, samples_per_file);
    
    for f=1:max(size(files))
        [raw_data, std_mean] = parseData(files(f).name);
        
        for row = 1:samples_per_file
            for stack = 1:num_stacks
                dataX(stack, :, (f-1) * samples_per_file + row) = [raw_data(row + stack - 1, :) 1];
                dataY(stack, :, (f-1) * samples_per_file + row) = raw_data(row + stack, :);
            end
        end
    end
    
    num_rows = size(dataX,3);
    
    i_perm = randperm(num_rows);
    
    data = struct;
    data.trainX = dataX(:, :, i_perm(1 : floor(num_rows / 2)));
    data.trainY = dataY(:, :, i_perm(1 : floor(num_rows / 2)));
    data.validateX = dataX(:, :, i_perm(floor(num_rows / 2) + 1 : floor(3 * num_rows / 4)));
    data.validateY = dataY(:, :, i_perm(floor(num_rows / 2) + 1 : floor(3 * num_rows / 4)));
    data.testX = dataX(:, :, i_perm(floor(3 * num_rows / 4) + 1 : num_rows));
    data.testY = dataY(:, :, i_perm(floor(3 * num_rows / 4) + 1 : num_rows));
end