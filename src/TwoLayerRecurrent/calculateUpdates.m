% Use the delta values and signals from each layer to update it's weight
% matrix. Sum the update values across all stacks of the network.

% Update values accumulated thus far
% Uinput = [d+1 x N] where d+1 is the number of input features + a bias 
% Uinterior = Uprev1 = Uprev2 = [N+1 x N] (same size but different values)
% Uoutput = [N+1 x l] where l is the nubmer of output features

% DjN, DiN and DpN are delta values for each layer of neurons (see
% backpropagation file)

% signals1, signals2 = signals entering the first and second hidden layer
% respectively from the same layer

% signals1prev, signals2prev = signals entering the first and second hidden
% layer respectively from the previous hidden layer

% X = the input values for each stack

% Output:
% Newly updated update values for weight matrices
% Uinput = [d+1 x N] where d+1 is the number of input features + a bias 
% Uinterior = Uprev1 = Uprev2 = [N+1 x N] (same size but different values)
% Uoutput = [N+1 x l] where l is the nubmer of output features

function [Uinput, Uinterior, Uprev1, Uprev2, Uoutput] = calculateUpdates(Uinput, Uinterior, Uprev1, Uprev2, Uoutput, X, signals1, signals1prev, signals2, signals2prev, DjN, DiN, DpN)
    Uinput = Uinput + X' * DpN(:,1:end-1);
    Uinterior = Uinterior + signals1' * DiN(:, 1:end-1);
    Uprev1 = Uprev1 + signals1prev' * DpN(:, 1:end-1);
    Uprev2 = Uprev2 + signals2prev' * DiN(:, 1:end-1);
    Uoutput = Uoutput + signals2' * DjN;
end