function [psiEst_t] = extended_kalman_filt_aposteriori_blkmodel(yVec, F, n_abVec, numClasses, ekfParams)

% [psiEst_t] = extended_kalman_filt_apriori_blkmodel(yVec, F, classSizeList, ekfParams)

% % >>>> for debug begin <<<<
% numSnapShots = 1e2;
% [synNet] = synthetic_blkmodel_gen_params_init();
% dbg = [];
% [W, ~, classLabelList, numClasses, dbg] = synthetic_blkmodel_gen(numSnapShots, synNet, dbg);
% Params.GammaMat = dbg.GammaMat;
% Params.GammaMat0 = dbg.GammaMat0;
% Params.muZero = synNet.muZero;
% 
% ekfParams.GammaMat = Params.GammaMat;
% ekfParams.GammaMatZero = Params.GammaMat0;
% ekfParams.muZeroVals = Params.muZero;
% 
% [yVec, m_abVec, n_abVec] = get_observation_vec_aposteriori(W(:,:,1), classLabelList(:,1), synNet.numClasses);
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
% Sigma matrix - observation noise variance computation
if ekfParams.simNetwork == 1
    % for synthetic network
    sig_abSq = yVec(:, indSnapShot).*(1-yVec(:, indSnapShot))./n_abVec(:, indSnapShot);
    SigMat = diag(sig_abSq);
else
    % real network to address the issue of the absence of edges and the
    % consequence of ill-conditioned matrix
    if indSnapShot == 1
        sig_abSq = yVec(:, indSnapShot).*(1-yVec(:, indSnapShot))./n_abVec(:, indSnapShot);
        % mention this in the report - notifiable change
        minVarVal = 1e-2;
        if sum(sig_abSq < minVarVal) > 0
            sig_abSq(sig_abSq < minVarVal) = minVarVal;
            sig_abSq(sig_abSq == inf) = minVarVal;
            SigMat = diag(sig_abSq);
        end
    end
end

% Kalman filter equations
[psiEst, REst] = extended_kalman_filter(yVec, psiEst, F, REst, GammaMat, SigMat); %n_abVec(:, indSnapShot));
psiEst_t = psiEst; %


%figure(1);subplot(131);plot(sigmoid_fun(psiEt_t.'));subplot(132);plot(yVec.');subplot(133);plot(yVec.'-sigmoid_fun(psiEst_t.'));


