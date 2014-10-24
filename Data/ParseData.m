[A date time D E F visibility H I J K L temperature N O P Q R S T dewpoint V W X windspeed Z windDir AB AC AD pressure AF AG AH AI AJ AK AL AM AN AO AP AQ AR] = textread('Seattle_WA.txt', '%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s', 'delimiter', ',');

date = str2double(date); %Date
time = str2double(time); %Time

%TO DO: ONLY ADD THE ROWS THAT ARE THE FIRST TIME TO OCCUR THAT HOUR, I.E.
%WHENEVER THE LAST TWO DIGITS OF TIME ROLL OVER.

%%time - floor(time/1000)*100 < 

visibility = str2double(visibility); %Visibility
temperature = str2double(temperature); %Temperature
dewpoint = str2double(dewpoint); %DewPointCelcius
windspeed = str2double(windspeed); %WindSpeed
windDir = str2double(windDir); %WindDirection
pressure = str2double(pressure); %StationPressure

temp = [visibility temperature dewpoint windspeed windDir pressure];
temp_nan = [isnan(visibility) isnan(temperature) isnan(dewpoint) isnan(windspeed) isnan(windDir) isnan(pressure)];
for i=1:max(size(temp(:,1)))
    for j=1:max(size(temp(1,:)))
        if temp_nan(i,j) == 1
            temp(i,j)=temp(i-1,j);
        end
    end
end
Matr = [visibility temperature dewpoint windspeed windDir pressure];