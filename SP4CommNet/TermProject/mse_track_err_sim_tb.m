% test bench to evaluate MSE tracking error performance for simulated experiment

numIters = 1; % number of iterations
burnInPeriod = 20; % in terms of number of snapshots
numSnapShots = 40;

% enable for [apriori blockmodel, aposteriori blockmodel, static stochastic blockmodel] - respectively
blockEnableVec = [0 1 0];

% initialize the parameters for time evolving dynamic SBM
[synNet] = synthetic_blkmodel_gen_params_init();
mseAprioriVec = zeros(1, numIters);
mseAposterioriVec = zeros(1, numIters);
mseSSBMVec = zeros(1, numIters);
psiEst_tSSBM3DTmp = zeros(synNet.numClasses, synNet.numClasses, numSnapShots);

for iterLoop = 1:numIters % loop for all iterations
    iterLoop
    dbg = [];
    % Generation of Stochastic Blockmodel
    % get the adjacency matrix and class labels for SBM
    [W, psiVec_t, classLabelList, numClasses, dbg] = synthetic_blkmodel_gen(numSnapShots, synNet, dbg);
    ParamsApriori.GammaMat = dbg.GammaMat;
    ParamsApriori.GammaMat0 = dbg.GammaMat0;
    ParamsApriori.muZero = synNet.muZero;
    
    if blockEnableVec(1) == 1
        %disp('Apriori Blockmodel');
        % get the apriori SBM
        [psiEst_tApriori, yVecApriori] = get_apriori_blkmodel(W, classLabelList, numClasses, ParamsApriori);
        
        % computing tracking error for apriori SBM - this should be straight forward
        % compute error
        errAprioriMat = psiEst_tApriori-logit_fun(yVecApriori);
        % discard the elements corresponding to the burn-in period
        errAprioriMat(:, 1:burnInPeriod) = 0;
        % compute MSE
        mseApriori = sum(sum(abs(errAprioriMat).^2))/((numSnapShots-burnInPeriod)*(synNet.numNodes^2));
        % collect MSEs in a vector
        mseAprioriVec(iterLoop) = mseApriori;
    end
    
    if blockEnableVec(2) == 1
        disp('Aposteriori Blockmodel');
        % get the aposteriori SBM
        ParamsAposteriori.GammaMat = dbg.GammaMat;
        % get posteriori blockmodel
        [psiEst_tAposteriori, c_tAposteriori, yVecAposteriori] = get_aposteriori_blkmodel(W, numClasses, ParamsAposteriori);
        % computing tracking error for aposteriori SBM - this should be straight forward
        % compute error - DO NOT USE psiVec_t - this is ideal one and not the simulated guy
        % This is only for performance comparison - classLabelList should not be input to any posterior BM block
        [yVec_t, ~, ~] = get_observation_vec_apriori(W, classLabelList, numClasses);
        errAposterioriMat = psiEst_tAposteriori-logit_fun(yVec_t);
        % discard the elements corresponding to the burn-in period
        errAposterioriMat(:, 1:burnInPeriod) = 0;
        % compute MSE
        mseAposteriori = sum(sum(abs(errAposterioriMat).^2))/((numSnapShots-burnInPeriod)*(synNet.numNodes^2));
        % collect MSEs in a vector
        mseAposterioriVec(iterLoop) = mseAposteriori;
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
        psiVec_t3D = reshape(psiVec_t, numClasses, numClasses, []);
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
        end % for indSnapShot
        errMat = psiEst_tSSBMOptim - psiVec_t;
        sumSqErr = sum(sum(errMat.^2));
        mseSSBM = sumSqErr/((numSnapShots-burnInPeriod)*(synNet.numNodes^2));
        % collect MSEs in a vector
        mseSSBMVec(iterLoop) = mseSSBM;
    end
    
end % for iterLoop

if blockEnableVec(1) == 1
    figure(1);hist(mseAprioriVec, 20);
    meanMse = mean(mseAprioriVec);
    stdMse = std(mseAprioriVec);
end
%figure(2);hist(mseAposterioriVec, 10);
%figure(3);hist(mseSSBMVec, 10);

cd('./simResults');
% Apriori BM
fileName = ['aprioriBM_numIters' num2str(numIters) '_burnInPeriod' num2str(burnInPeriod) '_numSnapShots' num2str(numSnapShots)];
save([fileName '.mat'], 'mseAprioriVec', 'synNet', 'ParamsApriori','meanMse','stdMse');
cd('../');

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

