function [Params] = sim_blkmodel_params_init(Params)

Params.simNetwork = 0; % 0 -> for real network (Ex. Enron email network or MIT reality mining)
                       % 1 -> for synthetic network

% Gen
Params.classEstAccuracyEnable = 0;

% Apriori Blockmodel
Params.apriori.eigEnable = 0;

% Aposteriori Blockmodel


% Static Stochastic Blockmodel


