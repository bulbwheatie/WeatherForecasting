function [errors] = get_combo_error(Winput, Winterior, Wprev1, Wprev2, Woutput, Winputa, Winteriora, Wprev1a, Wprev2a, Woutputa)
    [data, std_mean] = getData_struct('all', 8);
    errors = zeros(6,1);
    for j=1:6
        tmp_error = 0;
        for i=1:size(data.validateX, 3)
            [temp_y, ~, ~] = feedForward(data.validateX(:,:,i), Winput(:,:,j), Winterior(:,:,j), Wprev1(:,:,j), Wprev2(:,:,j), Woutput(:,:,j));
            [temp_ya, ~, ~] = feedForward(data.validateX(:,:,i), Winputa(:,:,j), Winteriora(:,:,j), Wprev1a(:,:,j), Wprev2a(:,:,j), Woutputa(:,:,j));
            Ypred = (temp_y + temp_ya)/2;
            tmp_error = tmp_error + sum((Ypred(end,:) - data.validateY(end,j,i)).^2);
        end
        errors(j,1) = tmp_error/size(data.validateX,3);
    end
end