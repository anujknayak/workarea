% load MIT reality data
% MIT reality mining data - preprocessing

% select valid persons
filCnt = 1;
for filInd = 1:length(s)
    if length(s(filInd).my_affil) == 0
        continue;
    else
        sFil(filCnt) = s(filInd);
        filCnt = filCnt + 1;
    end
end

% start date in the required format
dateTimeStart = datetime('07/01/2004','InputFormat','MM/dd/yyyy');
% This is required to get the day and/or week count of the events occured in the year 2015
numDaysIn2004 = datevec(datetime('12/31/2004')-dateTimeStart);

% get device macs
numPersons = length(sFil);
for indPerson = 1:numPersons
    allMacs{indPerson} = sFil(indPerson).my_mac{1};
end

% put the data in week structure
% loop for all persons
% determine the number of scans for the selected person
% loop for all scans
% determine week number for the particular scan
% update in week structure
%
weekStruct = [];
idxCntMat = [];

for indPerson = 1:numPersons
    indPerson
    numScans = length(sFil(indPerson).device_macs);
    for indScan = 1:numScans
        if length(sFil(indPerson).device_date) == 0
            continue;
        end
        % determine week number for the particular scan
        % convert the timestamp of the current scan to the desired format - datetime datatype
        dateStrCurrent = datestr(sFil(indPerson).device_date(indScan));
        if isequal(dateStrCurrent, '00-Jan-0000')
            continue;
        else
            dateTimeCurrent = datetime(dateStrCurrent);
        end
        dateDiffVec = datevec(dateTimeCurrent - dateTimeStart).';
        % >>>> for debug begin
        if dateDiffVec(3) > 100
            brkpnt1 = 1;
        end
        if isequal(dateStrCurrent(8+[0:3]), '2005')
            brkpnt1 = 1;
        end
        % >>>> for debug end
        % check if 2004 or 2005
        numDaysTillScan = dateDiffVec(3);
        if numDaysTillScan < 0
            continue;
        end
        if  floor(numDaysTillScan/7) >= length(weekStruct)
            weekStruct(floor(numDaysTillScan/7)+1).person(indPerson).deviceMacs{1} = sFil(indPerson).device_macs{indScan};
        else
            if indPerson > length(weekStruct(floor(numDaysTillScan/7)+1).person)
                weekStruct(floor(numDaysTillScan/7)+1).person(indPerson).deviceMacs{1} = sFil(indPerson).device_macs{indScan};
            else
                if length(weekStruct(floor(numDaysTillScan/7)+1).person(indPerson).deviceMacs) == 0
                    weekStruct(floor(numDaysTillScan/7)+1).person(indPerson).deviceMacs{1} = sFil(indPerson).device_macs{indScan};
                else
                    weekStruct(floor(numDaysTillScan/7)+1).person(indPerson).deviceMacs{end+1} = sFil(indPerson).device_macs{indScan};
                end
            end
        end
    end
end

