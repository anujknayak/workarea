% inputs: F     : state transition matrix
%         y     : observation
%         sigma : process noise
%         classSizesList : list of class sizes for all snapshots
%         ekfParams : 
% outputs: stateEst: estimated state
%
function [psiEst_t] = extended_kalman_filt_apriori_blkmodel(yVec, F, classSizeList, ekfParams)

% % >>>> for debug begin <<<<
% numSnapShots = 1e3;
% [synNet] = synthetic_blkmodel_gen_params_init();
% dbg = [];
% [~, W, ~, classSizeList, ~, dbg] = synthetic_blkmodel_gen(numSnapShots, synNet, dbg);
% Params.vCovMtx = dbg.vCovMtx;
% Params.vCovMtx0 = dbg.vCovMtx0;
% Params.muZero = synNet.muZero;
% 
% ekfParams.GammaMat = Params.vCovMtx;
% ekfParams.GammaMatZero = Params.vCovMtx0;
% ekfParams.muZeroVals = Params.muZero;
% 
% [yVec] = get_observation_vec_apriori(W, classSizeList);
% 
% F = eye(size(classSizeList,1)^2);
% % >>>> for debug end <<<<

numSnapShots = size(yVec, 2);
numClasses = size(classSizeList,1);

% Initialization
GammaMat = ekfParams.GammaMat;
% consider determining the hyperparameter - muZero
muZeroVals = ekfParams.muZeroVals;
muZeroMat = muZeroVals(2)*ones(numClasses); muZeroMat([1:numClasses:numClasses^2]+[0:1:numClasses-1]) = muZeroVals(1)*ones(1,numClasses);
muZeroVec = reshape(muZeroMat, [], 1);
psiEst = mvnrnd(muZeroVec, GammaMat, 1).';
psiEst_t = zeros(numClasses^2, numSnapShots);
% Initializing covariance estimate
REstTmp = zeros(numClasses^2);
for indSnapShot = 1:numSnapShots
    REstTmp = REstTmp + yVec(:, indSnapShot)*yVec(:, indSnapShot).';
end
REst = REstTmp/numSnapShots;

% Kalman filter loop
for indSnapShot = 1:numSnapShots
    % Sigma matrix - observation noise variance computation
    n_ab = classSizeList(:, indSnapShot)*classSizeList(:, indSnapShot).'; 
    n_ab = n_ab - eye(size(n_ab, 1));
    n_ab = n_ab(:);
    sig_abSq = yVec(:, indSnapShot).*(1-yVec(:, indSnapShot))./n_ab;
    SigMat = diag(sig_abSq);
    % Kalman filter equations
    [psiEst, REst] = extended_kalman_filter(yVec(:, indSnapShot), psiEst, F, REst, GammaMat, SigMat);
    psiEst_t(:, indSnapShot) = psiEst; % 
end

%figure(1);subplot(131);plot(sigmoid_fun(psiEst_t.'));subplot(132);plot(yVec.');subplot(133);plot(yVec.'-sigmoid_fun(psiEst_t.'));

