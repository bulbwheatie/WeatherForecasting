function data = getData()
    files = dir('../Data/*.txt');
        
    NUM_X_HOURS = 72;
    NUM_Y_HOURS = 24;
    exampleX = [];
    exampleY = [];
    
    for f=1:max(size(files))
        raw_data = parseData(files(f).name);
    
        %data is an n x m matrix
        n = max(size(raw_data(:,1)));
        m = max(size(raw_data(1,:)));

        tempX = zeros(n - NUM_X_HOURS - NUM_Y_HOURS, NUM_X_HOURS*m);
        tempY = zeros(n - NUM_X_HOURS - NUM_Y_HOURS, NUM_Y_HOURS*m);
        ind_example = 1;
        for i=NUM_X_HOURS:n-NUM_Y_HOURS

            row_X = zeros(1, NUM_X_HOURS*m);
            ind_row = 0;
            for j=i-NUM_X_HOURS+1:i
                row_X(1, (ind_row*m)+1:(ind_row+1)*m) = raw_data(j, :);
                ind_row = ind_row + 1;
            end
            tempX(ind_example,:) = row_X;

            row_Y = zeros(1, NUM_Y_HOURS*m);
            ind_row = 0;
            for j=i+1:i+NUM_Y_HOURS
                row_Y(1, (ind_row*m)+1:(ind_row+1)*m) = raw_data(j, :);
                ind_row = ind_row + 1;
            end
            tempY(ind_example,:) = row_Y;

            ind_example = ind_example + 1;
        end
        
        exampleX = [exampleX ; tempX];
        exampleY = [exampleY ; tempY];
    end
    
    num_rows = max(size(exampleX(:,1)));
    
    i_perm = randperm(num_rows);
    
    data = struct;
    data.trainX = exampleX(i_perm(1 : floor(num_rows / 2)), :);
    data.trainY = exampleY(i_perm(1 : floor(num_rows / 2)), :);
    data.validateX = exampleX(i_perm(floor(num_rows / 2) + 1 : floor(3 * num_rows / 4)), :);
    data.validateY = exampleY(i_perm(floor(num_rows / 2) + 1 : floor(3 * num_rows / 4)), :);
    data.testX = exampleX(i_perm(floor(3 * num_rows / 4) + 1 : num_rows), :);
    data.testY = exampleY(i_perm(floor(3 * num_rows / 4) + 1 : num_rows), :);
end