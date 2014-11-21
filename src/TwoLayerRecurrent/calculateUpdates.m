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
    d = size(Uinput, 1);
    s = size(signals1, 1); %number of stacks
    Ni = size(Uprev1, 1); %number of outgoing neurons
    Nj = size(Uprev1, 2); %number of incoming neurons
    l = size(Uoutput, 2); %number of features in the output
    
    % Calculate update the weights based on these deltas
%     for i=1:d
%         for j=1:Nj
%             tmpSum = 0;
%             for t=1:s
%                tmpSum = tmpSum + X(t,i) * DpN(t,j);
%             end
%             Uinput(i,j) = Uinput(i,j) + tmpSum;
%         end
%     end
    
    Uinput = Uinput + X' * DpN(:,1:end-1);
        
%     for i=1:Ni
%         for j=1:Nj
%             tmpSum = 0;
%             for t=1:s
%                tmpSum = tmpSum + signals1(t, i) * DiN(t,j);
%             end
%             Uinterior(i,j) = Uinterior(i,j) + tmpSum;
%         end
%     end

    Uinterior = Uinterior + signals1' * DiN(:, 1:end-1);
    
%     for i=1:Ni
%         for j=1:Nj
%             tmpSum = 0;
%             for t=2:s
%                tmpSum = tmpSum + signals1prev(t-1, i) * DpN(t,j);
%             end
%             Uprev1(i,j) = Uprev1(i,j) + tmpSum;
%         end
%     end
    
    Uprev1 = Uprev1 + signals1prev' * DpN(:, 1:end-1);

%     for i=1:Ni
%         for j=1:Nj
%             tmpSum = 0;
%             for t=2:s
%                tmpSum = tmpSum + signals2prev(t-1, i) * DiN(t,j);
%             end
%             Uprev2(i,j) = Uprev2(i,j) + tmpSum;
%         end
%     end

    Uprev2 = Uprev2 + signals2prev' * DiN(:, 1:end-1);
    
%     for i=1:Ni
%         for j=1:l
%             tmpSum = 0;
%             for t=1:s
%                tmpSum = tmpSum + signals2(t, i) * DjN(t,j);
%             end
%             Uoutput(i,j) = Uoutput(i,j) + tmpSum;
%         end
%     end
    Uoutput = Uoutput + signals2' * DjN;
end