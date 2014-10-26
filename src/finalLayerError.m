% Calculates the delta 

% Inputs
% X= d x 1 data sample with d features
% Ypred = d x 1 predicted output value
% Y = d x 1 actual output value
% W = d x p where p is number of perceptrons in layer



% Outputs
% delta = d x p values to update weight matrix 

function delta = finalLayerError(Ypred, Y)
    delta = (Ypred - Y) .* (1 - Ypred .* Ypred); 
    % delta = X * d.';
end