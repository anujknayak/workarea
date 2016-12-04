% test bench to evaluate MSE tracking error performance for simulated experiment

numIters = 50; % number of iterations
burnInPeriod = 10; % in terms of number of snapshots
numSnapShots = 20; % number of network time steps

% enable for [apriori blockmodel, aposteriori blockmodel, static stochastic blockmodel (aposteriori setting), static stochastic blockmodel (apriori setting)] - respectively
blockEnableVec = [1 0 0 1];

% initialize the parameters for time evolving dynamic SBM
[synNet] = synthetic_blkmodel_gen_params_init(); % parameters for synthetic BM generator
Params = [];[Params] = sim_blkmodel_params_init(Params); % parameters for SBM analysis

% Initialize vectors and matrices
mseAprioriVec = zeros(1, numIters); % MSE tracking error vector - Apriori SBM
mseAposterioriVec = zeros(1, numIters); % MSE tracking error vector - Aposteriori SBM
mseSSBMVec = zeros(1, numIters); % MSE tracking error vector - Static SBM (Aposteriori setting)
mseSSBMAprioriVec = zeros(1, numIters); % MSE tracking error vector - Static SBM (Apriori setting)
% psiEst - matrix (rather than a vector) handling is easier when shuffling of rows and colums is involved
% psiEst_tSSBM3DTmp - is used to align the block density vector (in sigmoid form) to the generated one to compute mean-squared tracking error
psiEst_tSSBM3DTmp = zeros(synNet.numClasses, synNet.numClasses, numSnapShots); 
eigMax1Mat = zeros(numSnapShots-burnInPeriod, numIters);
eigMax2Mat = zeros(numSnapShots-burnInPeriod, numIters);
AdjRandIndexAposterioriMat = zeros(numSnapShots, numIters);
AdjRandIndexSSBMMat = zeros(numSnapShots,numIters);

for iterLoop = 1:numIters % loop for all iterations
    iterLoop
    dbg = [];
    % Generation of Stochastic Blockmodel
    % get the adjacency matrix and class labels for SBM
    [W, psiVec_t, classLabelList, numClasses, dbg] = synthetic_blkmodel_gen(numSnapShots, synNet, dbg);
    
    if blockEnableVec(1) == 1
        %disp('Apriori Blockmodel');
        ParamsApriori = Params.apriori;
        ParamsApriori.GammaMat = dbg.GammaMat;
        %ParamsApriori.GammaMat0 = dbg.GammaMat0;
        %ParamsApriori.muZero = synNet.muZero;
        % get the apriori SBM
        [psiEst_tApriori, yVecApriori, ~, eigMax1Vec, eigMax2Vec] = get_apriori_blkmodel(W, classLabelList, numClasses, ParamsApriori);
        
        % computing tracking error for apriori SBM
        % compare with the ideal one - psiVec_t
        errAprioriMat = psiEst_tApriori-psiVec_t;
        % discard the elements corresponding to the burn-in period
        errAprioriMat(:, 1:burnInPeriod) = 0;
        % compute MSE
        mseApriori = sum(sum(abs(errAprioriMat).^2))/((numSnapShots-burnInPeriod)*(synNet.numNodes^2));
        % collect MSEs in a vector
        mseAprioriVec(iterLoop) = mseApriori;
        
        % comparing dominant eigen values of second order EKF term and observation noise matrix
        % discard eigen value samples corresponding to the burn-in period
        eigMax1Mat(:, iterLoop) = eigMax1Vec(burnInPeriod+1:end);
        eigMax2Mat(:, iterLoop) = eigMax2Vec(burnInPeriod+1:end);
        if Params.apriori.eigEnable == 1
            if iterLoop == numIters
                minEig1Vec = min(eigMax1Mat.').';
                maxhEig1Vec = max(eigMax2Mat.').';
                lowEig1Vec = min(eigMax1Mat.').' - mean(eigMax1Mat.').';
                highEig1Vec = max(eigMax1Mat.').' - mean(eigMax1Mat.').';
                lowEig2Vec = min(eigMax2Mat.').' - mean(eigMax2Mat.').';
                highEig2Vec = max(eigMax2Mat.').' - mean(eigMax2Mat.').';
                figure(6);
                errorbar([1:numSnapShots-burnInPeriod], mean(eigMax1Mat.').', lowEig1Vec, highEig1Vec, 'r', 'linewidth', 2);xlabel('Time step');ylabel('Eigenvalues');hold on;
                errorbar([1:numSnapShots-burnInPeriod], mean(eigMax2Mat.').', lowEig2Vec, highEig2Vec, 'b', 'linewidth', 2);xlabel('Time step');ylabel('Eigenvalues');grid on;
                xlim([1 numSnapShots-burnInPeriod]);
                title([num2str(synNet.numNodes) ' nodes']);
                set(gca,'fontsize', 20, 'YScale', 'log');hold off;
            end
        end
    end
    
    if blockEnableVec(2) == 1
        disp('Aposteriori Blockmodel');
        % get the aposteriori SBM
        ParamsAposteriori.GammaMat = dbg.GammaMat;
        % get posteriori blockmodel
        [psiEst_tAposteriori, c_tAposteriori, yVecAposteriori] = get_aposteriori_blkmodel(W, numClasses, ParamsAposteriori);
        % computing tracking error for aposteriori SBM
        % compare with the ideal one - psiVec_t
        errAposterioriMat = psiEst_tAposteriori - psiVec_t;
        % discard the elements corresponding to the burn-in period
        errAposterioriMat(:, 1:burnInPeriod) = 0;
        % compute MSE
        mseAposteriori = sum(sum(abs(errAposterioriMat).^2))/((numSnapShots-burnInPeriod)*(synNet.numNodes^2));
        % collect MSEs in a vector
        mseAposterioriVec(iterLoop) = mseAposteriori;
        %
        if Params.classEstAccuracyEnable == 1
            for indSnapShot = 1:numSnapShots
                [AdjRandIndexAposterioriMat(indSnapShot, iterLoop),~,~,~] = RandIndex(classLabelList(:, indSnapShot), c_tAposteriori(:, indSnapShot));
            end
        end
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
        %psiVec_t3D = reshape(psiVec_t, numClasses, numClasses, []);
        for indSnapShot = 1:numSnapShots
            % initialization
            indClassComboOptim = 0;
            sumErrBuf = inf;
            c_tSSBMCurrentSnapShot = c_tSSBM(:, indSnapShot);
            c_tSSBMTmp = zeros(synNet.numNodes, 1);
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
                [AdjRandIndexSSBMMat(indSnapShot, iterLoop),~,~,~] = RandIndex(classLabelList(:, indSnapShot), c_tSSBMOptim);
            end
        end % for indSnapShot
        % compute error - DO NOT USE psiVec_t - this is the ideal one and not the simulated guy
        % This is only for performance comparison - classLabelList should not be input to any posterior BM block
        % NOOO - you have to compare with the ideal one - psiVec_t
        %[yVec_t, ~, ~] = get_observation_vec_apriori(W, classLabelList, numClasses);
        %errMat = psiEst_tSSBMOptim - logit_fun(yVec_t);
        errMat = psiEst_tSSBMOptim - psiVec_t;
        sumSqErr = sum(sum(errMat.^2));
        mseSSBM = sumSqErr/((numSnapShots-burnInPeriod)*(synNet.numNodes^2));
        % collect MSEs in a vector
        mseSSBMVec(iterLoop) = mseSSBM;
    end
    
    if blockEnableVec(4) == 1
        [yVec_t, m_abVec, n_abVec] = get_observation_vec_apriori(W, classLabelList, numClasses);
        psiEst_tSSBMApriori = logit_fun(yVec_t);
        errMat = psiEst_tSSBMApriori - psiVec_t;
        sumSqErr = sum(sum(errMat.^2));
        mseSSBMApriori = sumSqErr/((numSnapShots-burnInPeriod)*(synNet.numNodes^2));
        % collect MSEs in a vector
        mseSSBMAprioriVec(iterLoop) = mseSSBMApriori;
    end
    
end % for iterLoop

% if blockEnableVec(1) == 1
%     figure(1);hist(mseAprioriVec, 20);
%     meanMse = mean(mseAprioriVec);
%     stdMse = std(mseAprioriVec);
% end
%
% if blockEnableVec(3) == 1
%     figure(1);hist(mseSSBMVec, 20);
%     meanMse = mean(mseSSBMVec);
%     stdMse = std(mseSSBMVec);
% end

if Params.classEstAccuracyEnable == 1
    AdjRandIndexSSBMVec = mean(AdjRandIndexSSBMMat);
    AdjRandIndexAposterioriVec = mean(AdjRandIndexAposterioriMat);
    AdjRandIndexMat = [AdjRandIndexSSBMVec;AdjRandIndexAposterioriVec].';
    figure(5);boxplot(AdjRandIndexMat);
end

if blockEnableVec(2) == 1 && blockEnableVec(3) == 1
    figure(6);boxplot([mseSSBMVec.' mseAposterioriVec.']);set(gca,'YScale','log');ylim([1e-6 1e-1]);
    title(['MSE tracking error performance - ' num2str(numIters) ' iterations']);ylabel('Mean-squared error');xlabel([num2str(synNet.numNodes) ' nodes']);
    set(gca, 'fontsize', 20, 'xticklabel', {'SSBM','Aposteriori EKF'});grid on;set(findobj(gca,'type','line'),'linew',2);
end

if blockEnableVec(1) == 1 && blockEnableVec(4) == 1
    figure(6);boxplot([mseSSBMAprioriVec.' mseAprioriVec.']);set(gca,'YScale','log');ylim([1e-6 1e-1]);
    title(['MSE tracking error performance - ' num2str(numIters) ' iterations']);ylabel('Mean-squared error');xlabel([num2str(synNet.numNodes) ' nodes']);
    set(gca, 'fontsize', 20, 'xticklabel', {'SSBM (Apriori)','Apriori EKF'});grid on;set(findobj(gca,'type','line'),'linew',2);
end

% cd('./simResults');
% if blockEnableVec(1) == 1
%     % Apriori BM
%     fileName = ['aprioriBM_numIters' num2str(numIters) '_burnInPeriod' num2str(burnInPeriod) '_numSnapShots' num2str(numSnapShots)];
%     save([fileName '.mat'], 'mseAprioriVec', 'synNet', 'ParamsApriori','meanMse','stdMse');
% end

% if blockEnableVec(3) == 1
%     % Apriori BM
%     fileName = ['ssbm_numIters' num2str(numIters) '_burnInPeriod' num2str(burnInPeriod) '_numSnapShots' num2str(numSnapShots)];
%     save([fileName '.mat'], 'mseAprioriVec', 'synNet','meanMse','stdMse');
% end

%cd('../');

% tracking error for aposteriori BM - shuffle class label configurations and iterate - find the class label for minimum error with the apriori BM - do this for every snapshot
% tracking error for SSBM - perform the operations similar to the aposterior BM


% CRAP - but the logic might be useful
% permc_tAll = perms([1:numClasses]);
% sumSqErrBuf = inf;
% for ind = 1:factorial(numClasses)
%     [~, c_tSSBMTmp] = ismember(c_tSSBM, permc_tAll(ind, :));
%     % assumption - class labels are labeled as [1 2 3 .. K]
%     errSSBMMat = psiEst_tSSBM-psiVec_t;
%     sumSqErrTmp = sum(sum(errSSBMMat.^2));
%     % discard the elements corresponding to the burn-in period
%     errSSBMMat(:, 1:burnInPeriod) = 0;
%     % compute MSE
%     if sumSqErrTmp < sumSqErrBuf
%         mseSSBM = sumSqErrTmp/((numSnapShots-burnInPeriod)*(synNet.numNodes^2));
%         sumSqErrBuf = sumSqErrTmp;
%         c_tSSBMUpdated = c_tSSBMTmp;
%     end
%     sumSqErrArr(ind) = sumSqErrTmp;
% end

