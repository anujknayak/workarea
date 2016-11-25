%function [psiEst_t] = get_aposteriori_blkmodel(W, Params)

% >>>> FOR DEBUG BEGIN <<<<
numSnapShots = 1e3;
[synNet] = synthetic_blkmodel_gen_params_init();
dbg = [];
[~, W, psiVec_t, ~, classSizeList, classMemNodeMap, dbg] = synthetic_blkmodel_gen(numSnapShots, synNet, dbg);
Params.vCovMtx = dbg.vCovMtx;
Params.vCovMtx0 = dbg.vCovMtx0;
Params.numClasses = synNet.numClasses;
Params.muZero = synNet.muZero;
% >>>> FOR DEBUG END <<<<

numClasses = Params.numClasses;
classListAll = [1:numClasses];
numNodes = size(W, 1);

% initialize class membership - spectral clustering
[c_0] = spectral_clustering(W(:,:,1), numClasses);

% compute block densities Y^t
[yVec, m_abVec, n_abVec] = get_observation_vec_aposteriori(W, c_0, numClasses);

% compute EKF equations
FMat = eye(size(classSizeList,1)^2);
ekfParams.GammaMat = Params.vCovMtx;
ekfParams.GammaMatZero = Params.vCovMtx0;
ekfParams.muZeroVals = Params.muZero;
[psiEst_t] = extended_kalman_filt_aposteriori_blkmodel(yVec, FMat, classSizeList, ekfParams);

% computing log posterior
% computing muZeroVec - consider determing this hyperparameter [TODO]
muZeroVals = ekfParams.muZeroVals;muZeroMat = muZeroVals(2)*ones(numClasses); muZeroMat([1:numClasses:numClasses^2]+[0:1:numClasses-1]) = muZeroVals(1)*ones(1,numClasses);
muZeroVec = reshape(muZeroMat, [], 1);
psiEstInit = mvnrnd(muZeroVec, ekfParams.GammaMat, 1).';
REstInit = FMat*ekfParams.GammaMatZero*FMat.' + ekfParams.GammaMat;
% log posterior
[pt] = posterior_prob_compute(m_abVec, n_abVec, psiEst_t, psiEstInit, REstInit);
% 
% c_tSorted = c_0Sorted;
% % Local search algorithm
% while iterLoop < maxIter
%     pt = -inf;
%     c_tTilda = c_tSorted;
%     for indNode = 1:numNodes
%         for indClass = 1:numClasses
%             if c_tSorted(indNode) == indClass
%                 continue;
%             else
%                 c_tTilda(indNode) = indClass;                
%                 % compute block densities
%                 [yVec, m_abVec, n_abVec] = get_observation_vec_aposteriori(W, c_tTilda);
%             end
%         end
%     end
% end
% 
% %figure(1);imagesc(WSorted);
% %figure(2);imagesc(W(:,:,1))






