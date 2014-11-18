function [data, std_mean] = parseData(textfile)

    NUM_FEATURES = 6;
    PATH = '../../data/';

    [time visibility temperature dewpoint windspeed windDir pressure] = textread(strcat(PATH, textfile), '%*s%*s%s%*s%*s%*s%s%*s%*s%*s%*s%*s%s%*s%*s%*s%*s%*s%*s%*s%s%*s%*s%*s%s%*s%s%*s%*s%*s%s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s', 'delimiter', ',');

    time = floor(str2double(time)/100); %this value is in the form hhmm, so get rid of the minutes by dividing by 100
    
    visibility = str2double(visibility); %Visibility(miles)
    temperature = str2double(temperature); %Temperature(celcius)
    dewpoint = str2double(dewpoint); %DewPointCelcius
    windspeed = str2double(windspeed); %WindSpeed(mph)
    windDir = str2double(windDir); %WindDirection
    pressure = str2double(pressure); %StationPressure

    %create the data matrix with one row for every 4th hour of the year
    data = zeros((24/4)*365, NUM_FEATURES);

    %only take the first row from each hour
    last = -1;
    new_row = 1;
    for old_row=1:max(size(time))
        if time(old_row) ~= last
            if (mod(time(old_row), 4) == 0)
                data(new_row,:) = [visibility(old_row) temperature(old_row) dewpoint(old_row) windspeed(old_row) windDir(old_row) pressure(old_row)];
                last = time(old_row);
                new_row = new_row + 1;
            end
        end
    end

    %get rid of bad data, i.e. NaN, and 0s for windspeed/wind direction
    for i=1:max(size(data(:,1)))
        for j=1:max(size(data(1,:)))
            %if the entry is not a number OR the entry is 0 and is from either windspeed or winddir, make it
            %equal to the previous entry
            if isnan(data(i,j)) == 1 || (data(i,j)==0 && (j==NUM_FEATURES-2 || j==NUM_FEATURES-1))
                if i ~= 1
                    data(i,j)=data(i-1,j);
                else
                    data(i,j)=0;
                end
            end
        end
    end
    
    %zero-mean the data, and divide by the std
    std_mean = zeros(2,NUM_FEATURES);
    std_mean(1,:) = std(data);
    std_mean(2,:) = mean(data);
    for i=1:NUM_FEATURES
        data(:,i) = data(:,i) - std_mean(2,i);
        data(:,i) = data(:,i) ./ std_mean(1,i);
    end
end