set(0,'defaultAxesFontSize',15)
%% Params list
MList = [1;2;4;8;10]; % list of # antennas
snrAvgdB = 10; % average SNR in dB
snrList = 0:1:60; % SNR list
snrAvg = 10^(snrAvgdB/10); % average SNR in linear scale
snrMat = ones(length(MList), 1)*snrList;
MMat = MList*ones(1,length(snrList));
%% pdf of SNR after selection combining
psigmaMatSelComb = (MMat)/snrAvg.*((1-exp(-(snrMat)/snrAvg)).^(MMat-1)).*(exp(-(snrMat)/snrAvg));
%% Plotting
figure(1);set(gca, 'FontSize', 20);
plotStruct = plot(snrList, psigmaMatSelComb.', 'linewidth', 2);xlabel('{\gamma}_{\Sigma}','FontSize', 15);ylabel('density');xlim([min(snrList) max(snrList)]);ylim([min(min(psigmaMatSelComb)) max(max(psigmaMatSelComb))]);
title('Selection Combining');
grid on;
markerList = {'x','o','^','*','s'};
for indCurve = 1:length(MList)
    plotStruct(indCurve).Marker = markerList{indCurve};
    legendStr{indCurve} = ['M = ' num2str(MList(indCurve))];
end
legend(legendStr);
%% pdf of SNR after MRC
psigmaMatMRC = ((snrMat).^(MMat-1).*exp(-(snrMat/snrAvg)))./((snrAvg.^(MMat)).*factorial(MMat-1));
%% Plotting
figure(2);set(gca, 'FontSize', 20);
plotStruct = plot(snrList, psigmaMatMRC.', 'linewidth', 2);xlabel('{\gamma}_{\Sigma}','FontSize', 15);ylabel('density');xlim([min(snrList) max(snrList)]);ylim([min(min(psigmaMatMRC)) max(max(psigmaMatMRC))]);
title('Maximal Ratio Combining');
grid on;
markerList = {'x','o','^','*','s'};
for indCurve = 1:length(MList)
    plotStruct(indCurve).Marker = markerList{indCurve};
    legendStr{indCurve} = ['M = ' num2str(MList(indCurve))];
end
legend(legendStr);

