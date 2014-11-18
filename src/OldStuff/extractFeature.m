function [new_predY new_realY] = extractFeature(predY, realY, feature)
    new_predY = zeros(1, 24);
    new_realY = zeros(1, 24);
    j = feature;
    for i=1:24
        new_predY(1, i) = predY(1, j);
        new_realY(1, i) = realY(1, j);
        j = j+6;
    end
end