function [] = plotAdjacencyMat(W, classLabels, timeIndex, figId)

if nargin <= 3
    figId = 1;
end

clrList = {'k','r','b','c','m','g'};
figure(figId);
numClassLabels = length(unique(classLabels));
for indClass = 1:numClassLabels
    nodeIndices = find(classLabels == indClass);
    % generate mask
    classMask = zeros(size(W));
    classMask(:, nodeIndices) = 1;
    classMask(nodeIndices, :) = 1;
    % plot
    WPlot = W.*classMask;
    spy(WPlot, clrList{indClass});hold on;
    legendList{indClass} = ['class ' num2str(indClass)];
end
xlabel('Node Number');ylabel('Node Number');
set(gca, 'fontsize', 18);
title(['Network at time index = ' num2str(timeIndex)]);
legend(legendList);

hold off;