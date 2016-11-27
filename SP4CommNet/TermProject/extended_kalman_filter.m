
function [psiEst, REst, psiEst_tMinus1, REst_tMinus1] = extended_kalman_filter(yVec, psiEst, FMat, REst, GammaMat, SigMat)

psiEst = FMat*psiEst; % predicting the next state
psiEst_tMinus1 = psiEst;
REst = FMat*REst*FMat.'+ GammaMat; % estimating the covariance matrix
REst_tMinus1 = REst;
HMat = diag((sigmoid_fun(psiEst).^2).*exp(-psiEst));
KalmanGainMat = (REst*HMat.')/(HMat*REst*HMat.'+SigMat); % computing the Kalman Gain
psiEst = psiEst + KalmanGainMat*(yVec-sigmoid_fun(psiEst)); % updating the state estimate
REst = (eye(size(KalmanGainMat)) - KalmanGainMat*HMat)*REst; % updating the covariance matrix estimate

%figure(1);imagesc(KalmanGainMat);drawnow();



