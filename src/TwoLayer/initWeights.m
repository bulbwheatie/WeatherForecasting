% Calculate weight of a specified size with random values in a specified
% range
function W = initWeights(row, col, min, max)
    W = (max-min).*rand(row, col) + min;
    %W(1,1) = 0;
end