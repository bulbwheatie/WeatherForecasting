function [Uinput, Uinterior, Uprev1, Uprev2, Uoutput] = calculateUpdates(Uinput, Uinterior, Uprev1, Uprev2, Uoutput, X, signals1, signals2, DiN)
    d = size(Uinput, 1);
    s = size(signals1, 1); %number of stacks
    n = size(Uprev1, 1); %number of neurons
    l = size(Uoutput, 2); %number of features in the output
    
    % Calculate update the weights based on these deltas
    for i=1:d
        for j=1:n
            tmpSum = 0;
            for t=1:s
               tmpSum = tmpSum + X(t,i) * DiN(t,j);
            end
            Uinput(i,j) = Uinput(i,j) + tmpSum;
        end
    end
    
    Uinterior = zeros(n, n);
    for i=1:n
        for j=1:n
            tmpSum = 0;
            for t=1:s
               tmpSum = tmpSum + signals1(t, j) * DiN(t,i);
            end
            Uprev1(i,j) = Uprev1(i,j) + tmpSum;
        end
    end
    
    Uprev1 = zeros(n, n);
    for i=1:n
        for j=1:n
            tmpSum = 0;
            for t=2:s
               tmpSum = tmpSum + signals1(t-1, j) * DiN(t,i);
            end
            Uprev1(i,j) = Uprev1(i,j) + tmpSum;
        end
    end
    
    Uprev2 = zeros(n, n);
    for i=1:n
        for j=1:n
            tmpSum = 0;
            for t=2:s
               tmpSum = tmpSum + signals2(t-1, j) * DiN(t,i);
            end
            Uprev2(i,j) = Uprev2(i,j) + tmpSum;
        end
    end
    
    Uoutput = zeros(size(Uoutput));
    for i=1:n
        for j=1:l
            tmpSum = 0;
            for t=1:s
               tmpSum = tmpSum + signals2(t, i) * DiN(t,j);
            end
            Uoutput(i,j) = Uoutput(i,j) + tmpSum;
        end
    end
end