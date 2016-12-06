function [] = plot_network(W, classLabelList, snapShotNum, numClasses, figId, savePlot, fileName)

numNodes = size(W, 1);
WMat = W(:,:,snapShotNum);
classLabelList = classLabelList(:, snapShotNum);

figure(figId);
[wxindices, wyindices] = find(WMat);
plot(wxindices, wyindices, 'c.', 'markersize', 20);hold on;

if numClasses > 8
    error('Max limit reached. Update colorList');
end

colorList = {'r.','b.','k.','g.','m.','y.','rx','bx'};
for indClass = 1:numClasses
    classMembers = find(classLabelList == indClass);
    refMask = zeros(numNodes);
    refMask(classMembers, classMembers) = 1;
    WClass = refMask.*WMat;
    [xIndices, yIndices] = find(WClass);
    figure(figId);plot(xIndices, yIndices, colorList{indClass}, 'markersize', 20);hold on;
end
figure(figId);hold off;
xlabel(['time step ' num2str(snapShotNum)]);
set(gca, 'fontsize', 20);
set(gcf,'units','normalized','outerposition',[0 0 1 1]);

if savePlot == 1
    fileName = [fileName '_snapShot' num2str(snapShotNum)];
    print(fileName, '-dpng');
end



