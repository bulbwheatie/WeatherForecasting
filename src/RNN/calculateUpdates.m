function [Uinput, Uprev, Uoutput] = calculateUpdates(Uinput, Uprev, Uoutput, X, signals, DiN)
    s = size(signals, 1); %number of stacks
    n = size(Uprev, 1); %number of neurons

    % Calculate update the weights based on these deltas
    for i=1:size(Uinput, 1)
        for j=1:size(Uinput,2)
            tmpSum = 0;
            for t=1:s
               tmpSum = tmpSum + X(t,i) * DiN(t,j);
            end
            Uinput(i,j) = Uinput(i,j) + tmpSum;
        end
    end
    
    for i=1:size(Uinput, 2)
        for j=1:size(Uinput,2)
            tmpSum = 0;
            for t=2:s
               tmpSum = tmpSum + signals(t-1, j) * DiN(t,i);
            end
            Uprev(i,j) = Uprev(i,j) + tmpSum;
        end
    end
    
    for i=1:size(Uoutput, 1)
        for j=1:size(Uoutput,2)
            tmpSum = 0;
            for t=1:s
               tmpSum = tmpSum + signals(t, i) * DiN(t,j);
            end
            Uoutput(i,j) = Uoutput(i,j) + tmpSum;
        end
    end

end