snrdBList = [10:1:30];
snrList = 10.^(snrdBList/10);
alphaM = 2;
betaM = 1;
M = 2.^[1:4];
markerList = {'-o', '-s', '-^', '--', '-x', '-*'};
colorList=lines(length(M));
ps = alphaM*(betaM/2).^-(M.'*ones(1,length(snrList))).*(ones(length(M),1)*snrList).^-(M.'*ones(1, length(snrList)));
for indM = 1:length(M)
    figure(1);semilogy(snrdBList, ps(indM, :), markerList{mod(indM-1, length(markerList))+1}, 'color', colorList(indM, :), 'linewidth', 2); ylim([10^-15 0.5]); hold on;
    legendList{indM} = ['M = ' num2str(M(indM))];
end
grid on;
xlabel('SNR (dB)');ylabel('P_s');legend(legendList);set(gca, 'fontsize', 20);
hold off;