
% initialize the parameters for time evolving dynamic SBM
[synNet] = synthetic_blkmodel_gen_params_init();

dbg = [];
% get the adjacency matrix and class labels for SBM
[W, Wsorted, classLabelList, classLabelListSorted, classSizeList, classMemNodeMap, dbg] = synthetic_blkmodel_gen(1000, synNet, dbg);
ParamsApriori.vCovMtx = dbg.vCovMtx;
ParamsApriori.vCovMtx0 = dbg.vCovMtx0;
ParamsApriori.muZero = synNet.muZero;

% get the apriori SBM
[psiEst_t] = get_apriori_blkmodel(Wsorted, classSizeList, ParamsApriori);

% get the aposteriori SBM
ParamsAposteriori.vCovMtx = dbg.vCovMtx;
ParamsAposteriori.vCovMtx0 = dbg.vCovMtx0;
% [] = get_aposteriori_blkmodel(W, ParamsAposteriori);
