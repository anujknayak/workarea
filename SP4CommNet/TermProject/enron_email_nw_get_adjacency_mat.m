% Enron email network
% function to form adjacency matrix
function [WMat, psiVec_t, yVec_t, classLabelList, numClasses] = enron_email_nw_get_adjacency_mat()

txtFileEnron = 'C:\Users\Anuj Nayak\Desktop\SigProc4CommNet\TermProject\EnronEmailNetwork\execs_email_linesnum.txt';
[timeStampList, fromList, toList] = textread(txtFileEnron);
% discarding invalid time samples
numInitSamplesDiscard = 174;
fromList(1:numInitSamplesDiscard) = [];
toList(1:numInitSamplesDiscard) = [];
timeStampList(1:numInitSamplesDiscard) = [];
numSec56Weeks = 60*60*24*7*56;
numInitSamples56Weeks = find(timeStampList < (numSec56Weeks+timeStampList(1)));

% Discarding first 56 weeks
fromList(numInitSamples56Weeks) = [];
toList(numInitSamples56Weeks) = [];
timeStampList(numInitSamples56Weeks) = [];

% Parameters - initialization
numWeeks = 120;
numNodes = 184;
numSecondsPerWeek = 60*60*24*7;
numClasses = 7;

% remove offset from time stamp list
timeStampList = timeStampList - timeStampList(1) + 1;
% pre-allocate size for the adjacency matrix
WMat = zeros(numNodes, numNodes, numWeeks);

for indWeek = 1:numWeeks
    % find indices corresponding to the current week
    indicesCurrentWeek = (timeStampList>(indWeek-1)*numSecondsPerWeek) & (timeStampList<=indWeek*numSecondsPerWeek);
    % extracting from and to values for the current week
    fromListCurrentWeek = fromList(indicesCurrentWeek);
    toListCurrentWeek = toList(indicesCurrentWeek);
    % number of time stamps for the current week
    % get the adjacency matrix for one week
    WMat((fromListCurrentWeek+1)+numNodes*(toListCurrentWeek) + (indWeek-1)*numNodes^2) = 1;
    numEmailsVec(indWeek) = length(fromListCurrentWeek);
end

% get class labels from Enron email network
[classLabelList] = enron_email_nw_get_class_label();

% initialize block density matrix
yVec_t = zeros(numClasses^2, numWeeks);

% get observation vector for 120 weeks
for indWeek = 1:numWeeks
    % obtain block density matrix for one week
    [yVec, m_abVec, n_abVec] = get_observation_vec_aposteriori(WMat(:, :, indWeek), classLabelList, numClasses);
    % accumulate the vector into the block density matrix
    yVec_t(:, indWeek) = yVec;
end

% class labels do not change with respect to time
classLabelList = repmat(classLabelList, 1, size(WMat, 3));
% logit of the block densities
psiVec_t = logit_fun(yVec_t);

%% FOR DEBUG
% figure(4);imagesc(reshape(yVec_t(:, 57), 7, 7), [0 0.6]);figure(7);imagesc(reshape(yVec_t(:, 88), 7, 7), [0 0.6]);
% figure(5);imagesc(reshape(yVec_t(:, 58), 7, 7), [0 0.6]);figure(8);imagesc(reshape(yVec_t(:, 89), 7, 7), [0 0.6]);
% figure(6);imagesc(reshape(yVec_t(:, 59), 7, 7), [0 0.6]);figure(9);imagesc(reshape(yVec_t(:, 90), 7, 7), [0 0.6]);
% figure(2);plot([1:numWeeks], numEmailsVec, 'linewidth', 2);xlabel('Week');ylabel('Number of emails');%xlim([1 120]);
% set(gca, 'fontsize', 20);title('Enron Email Network');grid on;

% poleVec = zeros(1,120);poleVec(112) = 4000;
% hold on;stem(poleVec, 'k');hold off;

% if max(numEmailsVec) < 3000
%     brkpnt1 = 1;
% end
%figure(1);plot(yVec_t([1:7]+7*1, :).');ylim([0 0.8]);
%figure(3);plot(yVec_t([3]+7*1, :).');ylim([0 0.8]); % CEOs to Presidents
%figure(1);plot(conv(yVec_t([5]+7*1, :).', 1/3*ones(1,3)));ylim([0 0.8]);
% % % % >>>> FOR DEBUG
% for ind = 1:7
% %for ind = 1:49
%     figure(1);plot(yVec_t([1:7]+7*(ind-1), :).');drawnow();
% %    ind
% %    figure(1);plot(yVec_t(ind, :).');
%     pause;
% end
