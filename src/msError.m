function err = msError(Ytarget, Ytest) 
    
    [row, col] = size(Ytest);
    err = 0;
    for j = 1: row
       err  = err + ((Ytest(j, 1) - Ytarget(j,1)) ^ 2);
    end
    err = err/row;
end