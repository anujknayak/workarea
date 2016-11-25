% inputs: W - adjacency matrix
%       : classLabelList - list of class labels for each snapshot in time
%       : classSizesList - list of class sizes corresponding to the labels
% outputs: psiEst_t: logit of edge probabilities
function [psiEst_t] = get_apriori_blkmodel(W, classSizeList, Params)

% % >>>> for debug begin <<<<
% numSnapShots = 1e3;
% [synNet] = synthetic_blkmodel_gen_params_init();
% dbg = [];
% [~, W, psiVec_t, classSizeList, classMemNodeMap, dbg] = synthetic_blkmodel_gen(numSnapShots, synNet, dbg);
% Params.vCovMtx = dbg.vCovMtx;
% Params.vCovMtx0 = dbg.vCovMtx0;
% Params.muZero = synNet.muZero;
% % >>>> for debug end <<<<

[yVec] = get_observation_vec_apriori(W, classSizeList);

FMat = eye(size(classSizeList,1)^2);
ekfParams.GammaMat = Params.vCovMtx;
ekfParams.GammaMatZero = Params.vCovMtx0;
ekfParams.muZeroVals = Params.muZero;
[psiEst_t] = extended_kalman_filt_apriori_blkmodel(yVec, FMat, classSizeList, ekfParams);



