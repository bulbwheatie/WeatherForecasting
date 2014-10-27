% Calculates the mean square error per sample

% Inputs
% Ytarget = m x d actual values of m samples with d features
% Ytest = m x d predicted values

% Outputs
% err = scalar - mean square error per sample
%

function err = msError(Ytarget, Ytest) 
    

    error = Ytarget - Ytest;
    error = error .^ 2;
    err = norm(error)/size(error, 1);
    
end