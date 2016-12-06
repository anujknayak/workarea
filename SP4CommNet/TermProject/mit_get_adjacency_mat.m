% function to generate adjacency matrix
function [WMat, classLabelList, numClasses] = mit_get_adjacency_mat()

load('./PreProcDataset/MITRealityMining/weekStructMacAddrClassLabel.mat');

allMacsVec = hex2dec(allMacs);
numClasses = length(unique(classLabelList));

numWeeks = length(weekStruct);
maxConnPerWeek = 30;

% loop for all weeks
% loop for all persons
% loop for all scans for each person
% get mac address indicator vectors for each scan
% 10 prominent connections
% set the corresponding entries in adjacency matrix to 1
WMat = zeros(length(allMacsVec), length(allMacsVec), numWeeks);

for indWeek = 1:numWeeks
    numPersons = length(weekStruct(indWeek).person);
    for indPerson = 1:numPersons
        numScans = length(weekStruct(indWeek).person(indPerson).deviceMacs);
        connIndicMat = [];
        for indScan = 1:numScans
            connIndicMat(indScan, :) = ismember(allMacsVec, weekStruct(indWeek).person(indPerson).deviceMacs{indScan});
        end
        if numScans > 0
            connStrengthVec = sum(connIndicMat);
            [connStrengthVecSorted, sortedIndices] = sort(connStrengthVec, 'descend');
            sortedIndices((connStrengthVecSorted == 0)) = [];
            if length(sortedIndices)<maxConnPerWeek
                selectFewSortedIndices = sortedIndices;
            else
                selectFewSortedIndices = sortedIndices(1:maxConnPerWeek);
            end
            WMat(indPerson, selectFewSortedIndices, indWeek) = 1;
        end
    end
end

