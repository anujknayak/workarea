% test bench to evaluate MSE tracking error performance for simulated experiment

numSnapShots = 120;

% enable for [apriori blockmodel, aposteriori blockmodel, static stochastic blockmodel] - respectively
blockEnableVec = [1 0 0];

% initialize the parameters for time evolving dynamic SBM
[synNet] = enron_email_nw_synthetic_blkmodel_gen_params_init();
Params = enron_email_nw_blkmodel_params_init([]);

% Initialization
psiEst_tSSBM3DTmp = zeros(synNet.numClasses, synNet.numClasses, numSnapShots);

dbg = [];
% Generation of Stochastic Blockmodel
% get the adjacency matrix and class labels for SBM
%[~, ~, ~, ~, dbg] = synthetic_blkmodel_gen(numSnapShots, synNet, dbg);
GammaMat = proc_noise_cov_mtx_gen(synNet.numClasses, synNet.gaussEvolCovParamVec(1), synNet.gaussEvolCovParamVec(2));

[W, psiVec_t, yVec_t, classLabelList, numClasses] = enron_email_nw_get_adjacency_mat();

ParamsApriori = Params.apriori;
ParamsApriori.GammaMat = GammaMat;
% get the apriori SBM
[psiEst_tApriori, yVecApriori, ~, ~, ~, ~, dbg] = get_apriori_blkmodel(W, classLabelList, numClasses, ParamsApriori);

% computing tracking error for apriori SBM - this should be straight forward
% compute error
errAprioriMat = psiEst_tApriori-logit_fun(yVecApriori);
% compute MSE
mseApriori = sum(sum(abs(errAprioriMat).^2))/((numSnapShots)*(synNet.numNodes^2));

% figure(1);plot(sigmoid_fun(psiEst_tApriori.'),'linewidth', 2);
% hold on;plot(yVec_t.');hold off;

figure(2);
% compute confidence interval vector pairs
ceoToPresEstVec = sigmoid_fun(psiEst_tApriori(16,:));
ceoToPresVec = sigmoid_fun(psiVec_t(16, :));
n_abCeoToPresVec = dbg.n_abVec(16, :);theta_abVec = ceoToPresEstVec./n_abCeoToPresVec;
confidenceVec = sqrt(theta_abVec.*(1-theta_abVec)./n_abCeoToPresVec);
xshade = [1:numSnapShots fliplr(1:numSnapShots)];
yshade = [ceoToPresEstVec+2*confidenceVec fliplr(ceoToPresEstVec-2*confidenceVec)];
fill(xshade,  yshade, [6 6 8]/8, 'edgecolor','none');hold on;
plot(ceoToPresEstVec.', 'b','linewidth', 2);
plot(ceoToPresVec.', 'b')
grid on;xlabel('Week');ylabel('Edge probability');ylim([0 0.5]);
hold off;

figure(3);
% compute confidence interval vector pairs
othToOthEstVec = sigmoid_fun(psiEst_tApriori(end-1,:));
othToOthVec = sigmoid_fun(psiVec_t(end-1, :));
n_abOthToOthVec = dbg.n_abVec(end, :);theta_abVec = othToOthEstVec./n_abOthToOthVec;
confidenceVec = sqrt(theta_abVec.*(1-theta_abVec)./n_abOthToOthVec);
xshade = [1:numSnapShots fliplr(1:numSnapShots)];
yshade = [othToOthEstVec+2*confidenceVec fliplr(othToOthEstVec-2*confidenceVec)];
fill(xshade,  yshade, [8 6 6]/8, 'edgecolor','none');hold on;
plot(othToOthEstVec.', 'r');
plot(othToOthVec.', 'r')
grid on;xlabel('Week');ylabel('Edge probability');%ylim([0 0.05]);
hold off;


