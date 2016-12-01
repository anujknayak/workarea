% Script to test the functionality of the modules

% enable for [apriori blockmodel, aposteriori blockmodel, static stochastic blockmodel] - respectively
blockEnableVec = [1 0 0];

% initialize the parameters for time evolving dynamic SBM
[synNet] = synthetic_blkmodel_gen_params_init();

dbg = [];
% Generation of Stochastic Blockmodel
% get the adjacency matrix and class labels for SBM
[W, psiVec_t, classLabelList, numClasses, dbg] = synthetic_blkmodel_gen(100, synNet, dbg);
ParamsApriori.GammaMat = dbg.GammaMat;
ParamsApriori.GammaMat0 = dbg.GammaMat0;
%ParamsApriori.muZero = synNet.muZero;

if blockEnableVec(1) == 1
    % get the apriori SBM
    [psiEst_tApriori, yVecEstApriori, yVecApriori] = get_apriori_blkmodel(W, classLabelList, numClasses, ParamsApriori);
end

if blockEnableVec(2) == 1
    % get the aposteriori SBM
    ParamsAposteriori.GammaMat = dbg.GammaMat;
    ParamsAposteriori.GammaMat0 = dbg.GammaMat0;
    ParamsAposteriori.muZero = synNet.muZero;
    % get posteriori blockmodel
    [psiEst_tAposteriori, c_tAposteriori] = get_aposteriori_blkmodel(W, numClasses, ParamsAposteriori);
end

if blockEnableVec(3) == 1
    % Static stochastic blockmodel
    [psiEst_tSSBM, c_tSSBM] = get_static_sbm(W, numClasses);
end

%figure(10);subplot(121);plot(sigmoid_fun(psiEst_t.'), 'linewidth', 2); subplot(122);plot(sigmoid_fun(psiEst_tAposteriorAll.'), 'linewidth', 2);grid on;
%figure(11);subplot(121);plot((psiEst_tApriori.'), 'linewidth', 2);ylim([min(min(psiEst_tApriori)) max(max(psiEst_tApriori))]);subplot(122);plot((psiEst_tAposteriori.'), 'linewidth', 2);ylim([min(min(psiEst_tApriori)) max(max(psiEst_tApriori))]);grid on;
%figure(12);subplot(121);plot(sigmoid_fun(psiVec_t.')); subplot(122);plot(sigmoid_fun(psiEst_tApriori.'), 'linewidth', 2);grid on;

figure(13);plot(sigmoid_fun(psiVec_t.')); hold on;plot(sigmoid_fun(psiEst_tApriori.'), 'linewidth', 2);grid on;hold off;




