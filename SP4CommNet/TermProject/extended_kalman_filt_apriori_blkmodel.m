% inputs: F     : state transition matrix
%         y     : observation
%         sigma : process noise
%         ekfParams : 
% outputs: stateEst: estimated state
%
function [psiEst_t, yVec, eigMax1Vec, eigMax2Vec, psiEst_tMinus1] = extended_kalman_filt_apriori_blkmodel(yVec, n_abVec, F, classLabelList, numClasses, ekfParams)
%dbstop if warning
% % >>>> for debug begin <<<<
% numSnapShots = 1e3;
% [synNet] = synthetic_blkmodel_gen_params_init();
% dbg = [];
% [W, psiVec_t, classLabelList, numClasses, dbg] = synthetic_blkmodel_gen(numSnapShots, synNet, dbg);
% Params.GammaMat = dbg.GammaMat;
% ekfParams.GammaMat = Params.GammaMat;
% [yVec, ~, n_abVec] = get_observation_vec_apriori(W, classLabelList, numClasses);
% F = eye(numClasses^2);
% % >>>> for debug end <<<<

numSnapShots = size(yVec, 2);

% Initialization
% Consider estimating the hyperparameter GammaMat
GammaMat = ekfParams.GammaMat;
psiEst_t = zeros(numClasses^2, numSnapShots);
psiEst_tMinus1 = zeros(numClasses^2, numSnapShots); % for determining hyperparameters - GammaMat - sdiag and snb -> Not used by anything else in Apriori BM case
% computing init hyperparameters
FMat = eye(numClasses^2);
% estimating muZero
muZeroVec = logit_fun(yVec(:, 1));
psiEst = muZeroVec;
% estimating Gamma0
sig_abSq = yVec(:, 1).*(1-yVec(:,1))./n_abVec(:,1);SigMat = diag(sig_abSq);
GMat = diag(1./yVec(:,1) + 1./(1-yVec(:,1)));
GammaMatZeroInit = GMat*SigMat*GMat.';
REst = FMat*GammaMatZeroInit*FMat.' + ekfParams.GammaMat;

% for performance
eigMax1Vec = zeros(1, numSnapShots);
eigMax2Vec = zeros(1, numSnapShots);

% Kalman filter loop
for indSnapShot = 1:numSnapShots
    %disp(indSnapShot);
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
                SigMat = diag(sig_abSq);
            end
        end
    end
    
    % Kalman filter equations
    [psiEst, REst, psiEstMinus1, REst_tMinus1] = extended_kalman_filter(yVec(:, indSnapShot), psiEst, F, REst, GammaMat, SigMat);
    psiEst_t(:, indSnapShot) = psiEst;
    psiEst_tMinus1(:, indSnapShot) = psiEstMinus1;
    
    % for performance - comparing second order EKF term with the
    % observation noise variance
    % get the second order EKF term
    if ekfParams.eigenEnable == 1
        [secOrEkfMat] = second_order_ekf_term_compute(psiEst, REst_tMinus1);
        % compute max eigen values for LHS and RHS
        [eigMax1, eigMax2] = eigen_val_compute(secOrEkfMat, SigMat);
        eigMax1Vec(indSnapShot) = eigMax1;
        eigMax2Vec(indSnapShot) = eigMax2;
    end
end

% Eigen value comparison
%figure(10);subplot(121);plot(yVec.');subplot(122);plot(sigmoid_fun(psiEst_t.'), 'linewidth', 2); grid on;%subplot(122);plot(yVec.'-sigmoid_fun(psiEst_t.'));

