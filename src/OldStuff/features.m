function B = features(X)
    % Normalize the input features by adding a 1 to the first column 
    [row, col] = size(X);
    T = ones(row, 1);
    B = [T,X]; 
end


