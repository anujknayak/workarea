% function for hyperparameter estimation for Enron email network
% Apriori blockmodel only

sdiagList = kron(10.^[-2:2], [2:4:10]);
snbList = kron(10.^[-3:-1], [2:4:10]);

burnInPeriod = 12; % in terms of number of snapshots
numSnapShots = 10;

% enable for [apriori blockmodel, aposteriori blockmodel, static stochastic blockmodel] - respectively
blockEnableVec = [1 0 0];

% initialize the parameters for time evolving dynamic SBM

Params = enron_email_nw_blkmodel_params_init([]);

%[W, classLabelList, numClasses] = mit_get_adjacency_mat();
load 'C:\Users\Anuj Nayak\Documents\MATLAB\SP4CommNet\TermProject\PreProcDataset\MITRealityMining\edgeMatClassLabel.mat';
W(:,:,1:burnInPeriod) = [];
W(:,:,numSnapShots+1:end) = [];

% HACK - refining - BEGIN
studList = [8 9]; % found manually
[studIndices] = ismember(classLabelList, studList);
classLabelList(studIndices) = 101;
classLabelList(~studIndices) = 102;
classLabelList = classLabelList - 100;
numClasses = length(unique(classLabelList));
% HACK - refining - END

classLabelList = repmat(classLabelList, 1, numSnapShots);
numNodes = size(W, 1);

for indsdiag = 1:length(sdiagList)
    for indsnb = 1:length(snbList)
        if sdiagList(indsdiag) < snbList(indsnb)
            continue;
        end
        disp([sdiagList(indsdiag) snbList(indsnb)]);
        GammaMat = proc_noise_cov_mtx_gen(numClasses, sdiagList(indsdiag), snbList(indsnb));
        ParamsApriori = Params.apriori;
        ParamsApriori.GammaMat = GammaMat;
        
        % >>>> FOR DEBUG BEGIN
        %ParamsApriori.GammaMat = rand(size(GammaMat));
        % >>>> FOR DEBUG END
        
        % get the apriori SBM
        [psiEst_tApriori, yEst_tApriori, yVec_t, ~, ~, psiEst_tMinus1Apriori] = get_apriori_blkmodel(W, classLabelList, numClasses, ParamsApriori);
        
        % computing tracking error for apriori SBM - this should be straight forward
        % compute error
        errAprioriMat = sigmoid_fun(psiEst_tMinus1Apriori)-yVec_t;
        % compute MSE
        mseApriori = sum(sum(abs(errAprioriMat).^2))/((numSnapShots)*(numNodes^2));
        % collect MSEs in a vector
        mseAprioriVec(indsdiag, indsnb) = mseApriori;
        %week89Vec = zeros(1,numSnapShots);week89Vec(89) = 1;
        %figure(1);plot((yVec_t(16, :).'));hold on;plot(sigmoid_fun(psiEst_tApriori(16,:)).','linewidth', 2);plot(week89Vec, 'k');hold off;drawnow();pause;
    end
end

%figure;surf(snbList, sdiagList, mseAprioriVec);xlabel('snb');ylabel('sdiag');set(gca, 'XScale', 'log');set(gca, 'YScale', 'log');set(gca, 'ZScale', 'log');
mseAprioriVec(mseAprioriVec == 0) = max(max(mseAprioriVec));
figure;surf(snbList,sdiagList,mseAprioriVec);xlabel('snb');ylabel('sdiag');set(gca, 'XScale', 'log');set(gca, 'YScale', 'log');
%figure;plot(sdiagList,mseAprioriVec);xlabel('sdiag');set(gca, 'XScale', 'log');set(gca, 'YScale', 'log');