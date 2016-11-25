% inputs: F     : state transition matrix
%         y     : observation
%         sigma : process noise
%         ekfParams : 
% outputs: stateEst: estimated state
%
function [psiEst_t] = extended_kalman_filt_apriori_blkmodel(yVec, n_abVec, F, classLabelList, numClasses, ekfParams)
% %dbstop if warning
% % >>>> for debug begin <<<<
% numSnapShots = 1e3;
% [synNet] = synthetic_blkmodel_gen_params_init();
% dbg = [];
% [W, psiVec_t, classLabelList, numClasses, dbg] = synthetic_blkmodel_gen(numSnapShots, synNet, dbg);
% Params.vCovMtx = dbg.vCovMtx;
% Params.vCovMtx0 = dbg.vCovMtx0;
% Params.muZero = synNet.muZero;
% 
% ekfParams.GammaMat = Params.vCovMtx;
% ekfParams.GammaMatZero = Params.vCovMtx0;
% ekfParams.muZeroVals = Params.muZero;
% 
% [yVec, ~, n_abVec] = get_observation_vec_apriori(W, classLabelList, numClasses);
% 
% F = eye(numClasses^2);
% % >>>> for debug end <<<<

numSnapShots = size(yVec, 2);

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
    disp(indSnapShot);
    % Sigma matrix - observation noise variance computation
    sig_abSq = yVec(:, indSnapShot).*(1-yVec(:, indSnapShot))./n_abVec(:, indSnapShot);
    SigMat = diag(sig_abSq);
    % Kalman filter equations
    [psiEst, REst] = extended_kalman_filter(yVec(:, indSnapShot), psiEst, F, REst, GammaMat, SigMat);
    psiEst_t(:, indSnapShot) = psiEst; % 
end

figure(10);subplot(121);plot(yVec.');subplot(122);plot(sigmoid_fun(psiEst_t.'), 'linewidth', 2); grid on;%subplot(122);plot(yVec.'-sigmoid_fun(psiEst_t.'));

