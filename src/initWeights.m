function W = initWeights(row, col, min, max)
    W = (max-min).*rand(row, col) + min;
    W(1,1) = 0;
end