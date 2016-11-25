%function [psiEst_tAll, c_tHatAll] = get_aposteriori_blkmodel(W, numClasses, Params)

% >>>> FOR DEBUG BEGIN <<<<
numSnapShots = 1e2;
[synNet] = synthetic_blkmodel_gen_params_init();
dbg = [];
[W, psiVec_t, classLabelList, numClasses, dbg] = synthetic_blkmodel_gen(numSnapShots, synNet, dbg);
Params.vCovMtx = dbg.vCovMtx;
Params.vCovMtx0 = dbg.vCovMtx0;
Params.numClasses = synNet.numClasses;
Params.muZero = synNet.muZero;
% >>>> FOR DEBUG END <<<<

maxIter = 20;

classListAll = [1:numClasses];
numNodes = size(W, 1);

% initialize class membership - spectral clustering
[c_0] = spectral_clustering(W(:,:,1), numClasses);
% compute block densities Y^t
[yVec, m_abVec, n_abVec] = get_observation_vec_aposteriori(W(:,:,1), c_0, numClasses);
yVecRef2 = yVec;

% compute EKF equations
FMat = eye(numClasses^2);
ekfParams.GammaMat = Params.vCovMtx;
ekfParams.GammaMatZero = Params.vCovMtx0;
ekfParams.muZeroVals = Params.muZero;

% computing muZeroVec - consider determing this hyperparameter [TODO]
muZeroVals = ekfParams.muZeroVals;muZeroMat = muZeroVals(2)*ones(numClasses); muZeroMat([1:numClasses:numClasses^2]+[0:1:numClasses-1]) = muZeroVals(1)*ones(1,numClasses);
muZeroVec = reshape(muZeroMat, [], 1);
psiEst_t = mvnrnd(muZeroVec, ekfParams.GammaMat, 1).';
REst_t = FMat*ekfParams.GammaMatZero*FMat.' + ekfParams.GammaMat;
c_tHat = c_0;

for indSnapShot = 1:numSnapShots
    fprintf('snapshot %d\n',indSnapShot);
    [yVec, m_abVec, n_abVec] = get_observation_vec_aposteriori(W(:,:,indSnapShot), c_tHat, numClasses);
    yVecBar = yVec;
    % Sigma matrix - observation noise variance computation
    sig_abSq = yVec.*(1-yVec)./n_abVec;
    SigMat = diag(sig_abSq);
    % Extended Kalman Filter
    [psiEst_t, REst_t, psiEst_tMinus1, REst_tMinus1] = extended_kalman_filter(yVec, psiEst_t, FMat, REst_t, ekfParams.GammaMat, SigMat);
    % compute log posterior
    [pt] = posterior_prob_compute(m_abVec, n_abVec, psiEst_t, psiEst_tMinus1, REst_tMinus1);

    iterLoop = 0;
    % Local search algorithm
    while iterLoop < maxIter
        ptBar = -inf;
        c_tTilda = c_tHat;
        ptArr = [];
        for indNode = 1:numNodes
            for indClass = 1:numClasses
                if c_tTilda(indNode) == indClass
                    continue;
                else
                    c_tTilda(indNode) = indClass;
                    % compute block densities
                    [yVec, m_abVec, n_abVec] = get_observation_vec_aposteriori(W(:,:,indSnapShot), c_tTilda, numClasses);
                    % EKF equations
                    % Sigma matrix - observation noise variance computation
                    sig_abSq = yVec.*(1-yVec)./n_abVec;
                    SigMat = diag(sig_abSq);
                    % Extended Kalman Filter
                    [psiEst_tTilda, REst_tTilda, psiEst_tMinus1, REst_tMinus1] = extended_kalman_filter(yVec, psiEst_t, FMat, REst_t, ekfParams.GammaMat, SigMat);                    
                    % compute posterior probability
                    [ptTilda] = posterior_prob_compute(m_abVec, n_abVec, psiEst_tTilda, psiEst_tMinus1, REst_tMinus1);
                    %ptArr = [ptArr ptTilda];
                    %figure(1);plot(ptArr, 'b-o');drawnow();
                    if ptTilda > ptBar % Visited solution is better than the best neighboring solution
                        ptBar = ptTilda;
                        psiEst_tBar = psiEst_tTilda;
                        c_tBar = c_tTilda;                        
                        REst_tBar = REst_t;
                        yVecBar = yVec;
                        % >>>> for debug begin <<<<
                        %psiEst_t = psiEst_tTilda;
                        % >>>> for debug end <<<<
                    end
                    % Reset the class membership of the current node
                    c_tTilda(indNode) = c_tHat(indNode);
                end
            end
        end
        if ptBar > pt % Best neighboring solution is better than the current best solution
            pt = ptBar;
            psiEst_t = psiEst_tBar;
            c_tHat = c_tBar;
            %fprintf('\t got a better solution\n');
        else % Reached local maximum
            disp(iterLoop);
            break;
        end
        iterLoop = iterLoop + 1;
    end
    psiEst_tAll(:, indSnapShot) = psiEst_t;
    c_tHatAll(:, indSnapShot) = c_tHat;
    yVecAll(:, indSnapShot) = yVecBar;
end


figure(1);plot(sigmoid_fun(psiEst_tAll.'), 'linewidth', 2);hold on;plot(yVecAll.');figure(2);plot(yVecAll.'-sigmoid_fun(psiEst_tAll.'));





