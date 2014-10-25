function W = initWeights(row, col, min, max)
    W = (max-min).*rand(row + 1, col) + min;
    W(1,1) = 0;
end