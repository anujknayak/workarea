% test bench to evaluate MSE tracking error performance for simulated experiment

numIters = 1; % number of iterations
burnInPeriod = 0; % in terms of number of snapshots
numSnapShots = 20; % number of network time steps

% enable for [apriori blockmodel, aposteriori blockmodel, static stochastic blockmodel (aposteriori setting), static stochastic blockmodel (apriori setting)] - respectively
blockEnableVec = [0 0 1 0];
plotEdgeMat = 0;

%[W, classLabelList, numClasses] = mit_get_adjacency_mat();
%load 'C:\Users\Anuj Nayak\Documents\MATLAB\SP4CommNet\TermProject\PreProcDataset\MITRealityMining\edgeMatClassLabel.mat';
load 'C:\Users\Anuj Nayak\Documents\MATLAB\SP4CommNet\TermProject\PreProcDataset\MITRealityMining\edgeMatClassLabelMaxConn20.mat';
%load 'C:\Users\Anuj Nayak\Documents\MATLAB\SP4CommNet\TermProject\PreProcDataset\MITRealityMining\edgeMatClassLabelMaxConn30.mat';

% HACK - refining - BEGIN
%     '1styeargrad '
%     'grad'
%     'mlfrosh'
%     'mlgrad'
%     'mlstaff'
%     'mlurop'
%     'professor'
%     'sloan'
%     'sloan_2'
studList = [8]; % found manually
[studIndices] = ismember(classLabelList, studList);
classLabelList(studIndices) = 101;
classLabelList(~studIndices) = 102;
classLabelList = classLabelList - 100;
numClasses = length(unique(classLabelList));
% HACK - refining - END

numWeeksDiscard = 12;
W = W(:,:,numWeeksDiscard+[1:numSnapShots]);

% for debug
%W = round(rand(size(W))*4/6);

classLabelList = repmat(classLabelList, 1, numSnapShots);

numNodes = size(W, 1);
GammaMat = proc_noise_cov_mtx_gen(numClasses, 2, 1);
% initialize the parameters for time evolving dynamic SBM
Params = [];[Params] = mit_blkmodel_params_init(Params); % parameters for SBM analysis
% 
% % Initialize vectors and matrices
% % psiEst - matrix (rather than a vector) handling is easier when shuffling of rows and colums is involved
% % psiEst_tSSBM3DTmp - is used to align the block density vector (in sigmoid form) to the generated one to compute mean-squared tracking error
% psiEst_tSSBM3DTmp = zeros(numClasses, numClasses, numSnapShots);
% AdjRandIndexAposterioriVec = zeros(1, numSnapShots);
% AdjRandIndexSSBMVec = zeros(1, numSnapShots);


% for debug
if plotEdgeMat == 1
    colorList = {'r.','b.','k.','g.','m.','c.','y.','rx','bx'};
    for indSnapShot = 1:numSnapShots
        for indClass = 1:numClasses
            classMembers = find(classLabelList(:,indSnapShot) == indClass);
            refMask = zeros(size(W(:,:,indSnapShot)));
            refMask(classMembers, :) = 1;
            refMask(:, classMembers) = 1;
            WClass = refMask.*W(:,:,indSnapShot);
            [xIndices, yIndices] = find(WClass);
            figure(102);plot(xIndices, yIndices, colorList{indClass}, 'markersize', 20);hold on;
            drawnow();
        end
        %pause;
        figure(102);hold off;
    end
end
% for debug

if blockEnableVec(1) == 1
    disp('Apriori Blockmodel');
    ParamsApriori = Params.apriori;
    ParamsApriori.GammaMat = GammaMat;
    % get the apriori SBM
    [psiEst_tApriori, yVecApriori] = get_apriori_blkmodel(W, classLabelList, numClasses, ParamsApriori);
% computing tracking error for apriori SBM
figure;plot(sigmoid_fun(psiEst_tApriori.'), '-s', 'linewidth', 2);hold on;plot(yVecApriori.');ylim([0 1]);
end

if blockEnableVec(2) == 1
    disp('Aposteriori Blockmodel');
    % get the aposteriori SBM
    ParamsAposteriori.GammaMat = GammaMat;
    % get posteriori blockmodel
    [psiEst_tAposteriori, c_tAposteriori, yVecAposteriori] = get_aposteriori_blkmodel(W, numClasses, ParamsAposteriori);
    
    if plotEdgeMat == 1
        colorList = {'r.','b.','k.','g.','m.','c.','y.','rx','bx'};
        for indSnapShot = 1:numSnapShots
            for indClass = 1:numClasses
                classMembers = find(classLabelList(:,indSnapShot) == indClass);
                refMask = zeros(size(W(:,:,indSnapShot)));
                refMask(classMembers, :) = 1;
                refMask(:, classMembers) = 1;
                WClass = refMask.*W(:,:,indSnapShot);
                [xIndices, yIndices] = find(WClass);
                figure(100);plot(xIndices, yIndices, colorList{indClass}, 'markersize', 20);hold on;
                classMembers = find(c_tAposteriori(:, indSnapShot) == indClass);
                refMask = zeros(size(W(:,:,indSnapShot)));
                refMask(classMembers, :) = 1;
                refMask(:, classMembers) = 1;
                WClass = refMask.*W(:,:,indSnapShot);
                [xIndicesEst, yIndicesEst] = find(WClass);
                figure(101);plot(xIndicesEst, yIndicesEst, colorList{indClass}, 'markersize', 20);hold on;
                drawnow();
            end
            %pause;
            figure(100);hold off;
            figure(101);hold off;
        end
    end
    
    % class estimation accuracy
    if Params.classEstAccuracyEnable == 1
        for indSnapShot = 1:numSnapShots
            [AdjRandIndexAposterioriVec(indSnapShot),~,~,~] = RandIndex(classLabelList(:, indSnapShot), c_tAposteriori(:, indSnapShot));
        end
    end
    figure(1);plot(sigmoid_fun(psiEst_tAposteriori.'), 'linewidth', 2);ylim([0 1]);
    disp('class estimation accuracy');
    disp(AdjRandIndexAposterioriVec);
end

if blockEnableVec(3) == 1
    disp('Static Stochastic Blockmodel')
    % Static stochastic blockmodel
    [psiEst_tSSBM, c_tSSBM] = get_static_sbm(W, numClasses);
    % computing tracking error for static SBM - this should be straight forward
    % compute error
    % initializations
    permc_tAll = perms([1:numClasses]);
    psiEst_tSSBMOptim = zeros(numClasses^2, numSnapShots);
    % reshaping for convenience
    psiEst_tSSBM3D = reshape(psiEst_tSSBM, numClasses, numClasses, []);
    for indSnapShot = 1:numSnapShots
        % initialization
        indClassComboOptim = 0;
        sumErrBuf = inf;
        c_tSSBMCurrentSnapShot = c_tSSBM(:, indSnapShot);
        c_tSSBMTmp = zeros(numNodes, 1);
        % instead of searching for min error in block densities, search
        % for min error in class estimate vectors
        for indClassCombo = 1:factorial(numClasses)
            % form a new class vector
            for indClass = 1:numClasses
                % assumptions - classes are labeled as [1 ... K]
                c_tSSBMTmp(c_tSSBMCurrentSnapShot == indClass, 1) = permc_tAll(indClassCombo, indClass);
            end
            % compute hamming distance between class vectors
            sumErrTmp = sum(c_tSSBMTmp ~= c_tSSBMCurrentSnapShot);
            if sumErrTmp < sumErrBuf
                sumErrBuf = sumErrTmp;
                indClassComboOptim = indClassCombo;
                c_tSSBMOptim = c_tSSBMTmp;
            end
        end
        psiEst_tSSBMOptim(:, indSnapShot) = reshape(psiEst_tSSBM3D(permc_tAll(indClassComboOptim, :), permc_tAll(indClassComboOptim, :), indSnapShot), [], 1);
        %permc_tAll(indClassComboOptim, :)
        %figure(1);plot(c_tSSBM(:, indSnapShot));
        %figure(1);imagesc(psiEst_tSSBM3DCurrentSnapShot);
        %figure(2);subplot(121);imagesc(psiEst_tSSBM3D(permc_tAll(indClassComboOptim, :), permc_tAll(indClassComboOptim, :), indSnapShot));
        %subplot(122);imagesc(reshape(psiVec_t(:, indSnapShot), numClasses, numClasses));
        
        if Params.classEstAccuracyEnable == 1
            % class estimation accuracy for aposteriori blockmodel
            [AdjRandIndexSSBMVec(indSnapShot),~,~,~] = RandIndex(classLabelList(:, indSnapShot), c_tSSBMOptim);
        end
    end % for indSnapShot
    figure(1);plot(sigmoid_fun(psiEst_tSSBMOptim.'), 'linewidth', 2);ylim([0 1]);
end

if blockEnableVec(4) == 1
    [yVec_t, m_abVec, n_abVec] = get_observation_vec_apriori(W, classLabelList, numClasses);
    figure(1);plot(yVec_t.', 'linewidth', 2);ylim([0 1]);
end

if Params.classEstAccuracyEnable == 1
    AdjRandIndexSSBM = mean(AdjRandIndexSSBMVec);
    AdjRandIndexAposteriori = mean(AdjRandIndexAposterioriVec);
end



