% Calculates the mean square error per sample

% Inputs
% Ytarget = 1 x d actual values of m samples with d features
% Ytest = 1 x d predicted values

% Outputs
% err = scalar - mean square error per sample
%

function err = msError(Ytarget, Ytest) 

    % err = (((Ytarget - Ytest)./Ytest) * transpose((Ytarget - Ytest)./Ytest))/size(Ytest, 2);
    err = (((Ytarget - Ytest)) * transpose((Ytarget - Ytest)))/size(Ytest, 2);
    
end