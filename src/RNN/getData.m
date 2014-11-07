function [data, std_mean] = getData()
    files = dir('../../data/*.txt');
    rows_per_file = (24/4)*365;
    NUM_FEATURES = 6;
    data = zeros(rows_per_file*max(size(files)), NUM_FEATURES);
    for f=1:max(size(files))
        [raw_data, std_mean] = parseData(files(f).name);
        data(((f-1)*rows_per_file)+1:f*rows_per_file,:) = raw_data;
    end
end