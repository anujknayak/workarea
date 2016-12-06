function [Params] = mit_blkmodel_params_init(Params)

Params.simNetwork = 0;

% Gen
Params.classEstAccuracyEnable = 1;

% Apriori Blockmodel
Params.apriori.simNetwork = Params.simNetwork;
Params.apriori.eigEnable = 0;

% Aposteriori Blockmodel


% Static Stochastic Blockmodel


