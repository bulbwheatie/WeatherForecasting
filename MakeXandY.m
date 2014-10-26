function [ExampleX ExampleY] = MakeXandY(Data)
    NUM_X_HOURS = 72;
    NUM_Y_HOURS = 24;
    
    %data is an n x m matrix
    n = max(size(Data(:,1)));
    m = max(size(Data(1,:)));
    
    ExampleX = zeros(n - NUM_X_HOURS - NUM_Y_HOURS, NUM_X_HOURS*m);
    ExampleY = zeros(n - NUM_X_HOURS - NUM_Y_HOURS, NUM_Y_HOURS*m);
    ind_example = 1;
    for i=NUM_X_HOURS:n-NUM_Y_HOURS
        
        row_X = zeros(1, NUM_X_HOURS*m);
        ind_row = 0;
        for j=i-NUM_X_HOURS+1:i
            row_X(1, (ind_row*m)+1:(ind_row+1)*m) = Data(j, :);
            ind_row = ind_row + 1;
        end
        ExampleX(ind_example,:) = row_X;
        
        row_Y = zeros(1, NUM_Y_HOURS*m);
        ind_row = 0;
        for j=i+1:i+NUM_Y_HOURS
            row_Y(1, (ind_row*m)+1:(ind_row+1)*m) = Data(j, :);
            ind_row = ind_row + 1;
        end
        ExampleY(ind_example,:) = row_Y;
        
        ind_example = ind_example + 1;
    end
end