% inputs: F     : state transition matrix
%         y     : observation
%         sigma : process noise
%         ekfParams : 
% outputs: stateEst: estimated state
%
function [psiEst_t, yVec] = extended_kalman_filt_apriori_blkmodel(yVec, n_abVec, F, classLabelList, numClasses, ekfParams)
%dbstop if warning
% % >>>> for debug begin <<<<
% numSnapShots = 1e3;
% [synNet] = synthetic_blkmodel_gen_params_init();
% dbg = [];
% [W, psiVec_t, classLabelList, numClasses, dbg] = synthetic_blkmodel_gen(numSnapShots, synNet, dbg);
% Params.GammaMat = dbg.GammaMat;
% Params.GammaMat0 = dbg.GammaMat0;
% Params.muZero = synNet.muZero;
% 
% ekfParams.GammaMat = Params.GammaMat;
% ekfParams.GammaMatZero = Params.GammaMat0;
% ekfParams.muZeroVals = Params.muZero;
% 
% [yVec, ~, n_abVec] = get_observation_vec_apriori(W, classLabelList, numClasses);
% 
% F = eye(numClasses^2);
% % >>>> for debug end <<<<

numSnapShots = size(yVec, 2);

% Initialization
% Consider estimating the hyperparameter GammaMat
GammaMat = ekfParams.GammaMat;
psiEst_t = zeros(numClasses^2, numSnapShots);
% computing init hyperparameters
FMat = eye(numClasses^2);
% estimating muZero
muZeroVec = sigmoid_fun(yVec(:, 1));
psiEst = muZeroVec;
% estimating Gamma0
sig_abSq = yVec(:, 1).*(1-yVec(:,1))./n_abVec(:,1);SigMat = diag(sig_abSq);
GMat = diag(1./yVec(:,1) + 1./(1-yVec(:,1)));
GammaMatZeroInit = GMat*SigMat*GMat.';
REst = FMat*GammaMatZeroInit*FMat.' + ekfParams.GammaMat;

% Kalman filter loop
for indSnapShot = 1:numSnapShots
    %disp(indSnapShot);
    % Sigma matrix - observation noise variance computation
    sig_abSq = yVec(:, indSnapShot).*(1-yVec(:, indSnapShot))./n_abVec(:, indSnapShot);
    SigMat = diag(sig_abSq);
    % Kalman filter equations
    [psiEst, REst] = extended_kalman_filter(yVec(:, indSnapShot), psiEst, F, REst, GammaMat, SigMat);
    psiEst_t(:, indSnapShot) = psiEst; % 
end

%figure(10);subplot(121);plot(yVec.');subplot(122);plot(sigmoid_fun(psiEst_t.'), 'linewidth', 2); grid on;%subplot(122);plot(yVec.'-sigmoid_fun(psiEst_t.'));

