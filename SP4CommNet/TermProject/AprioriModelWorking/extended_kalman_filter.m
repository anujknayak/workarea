
function [psiEst, REst] = extended_kalman_filter(yVec, psiEst, FMat, REst, GammaMat, SigMat)

psiEst = FMat*psiEst; % predicting the next state
REst = FMat*REst*FMat.'+ GammaMat; % estimating the covariance matrix
HMat = diag((sigmoid_fun(psiEst).^2).*exp(-psiEst));
KalmanGainMat = REst*HMat.'*inv(HMat*REst*HMat.'+SigMat); % computing the Kalman Gain
psiEst = psiEst + KalmanGainMat*(yVec-sigmoid_fun(psiEst)); % updating the state estimate
REst = (eye(size(KalmanGainMat)) - KalmanGainMat*HMat)*REst; % updating the covariance matrix estimate



