function [data, std_mean] = getData(mode)
    if strcmp(mode, 'all')
        files = dir('../../data/*.txt');
    elseif  strcmp(mode, 'test')
        files = dir('../../data/test/leb.txt');
    else
        files = dir('../../data/*WA.txt');
    end
    if strcmp(mode, 'test')
        %only one month of data
        rows_per_file = (24/4)*31;
    else
        rows_per_file = (24/4)*365;
    end
    NUM_FEATURES = 6;
    data = zeros(rows_per_file*max(size(files)), NUM_FEATURES);
    for f=1:max(size(files))
        [raw_data, std_mean] = parseData(files(f).name, mode);
        data(((f-1)*rows_per_file)+1:f*rows_per_file,:) = raw_data;
    end
end