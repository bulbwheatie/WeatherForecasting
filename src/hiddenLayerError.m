function delta = hiddenLayerError(deltaPrev, W, Xout)
    
    delta = zeros(size(W,1), size(W, 2));
    for i = 1:(size(W, 1))
        sum = 0;
        for j = 1:(size(deltaPrev, 1))
            sum = sum + deltaPrev(j ,1) * W(i, j);
        end
        delta(1, i) = sum * (1 - Xout(1,i) * Xout(1,i));
    end
end