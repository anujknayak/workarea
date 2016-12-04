function [synNet] = enron_email_nw_synthetic_blkmodel_gen_params_init()

% % % % % %% >>>> FOR DEBUG - DIFFERENT TEST PARAMETERS HERE
% Enron network parameters
synNet.gaussEvolCovParamVec = [1 0.2]; % Covariance matrix for Gaussian random walk model
synNet.numNodes = 184; % number of nodes in the network
synNet.numClasses = 7; % number of class labels
%synNet.classSizes = synNet.numNodes/synNet.numClasses*ones(1,synNet.numClasses);
%synNet.classMemVecInit = repelem([1:synNet.numClasses], synNet.classSizes).'; % different class sizes can also be given. However, the literature does not demand this.
%synNet.classMemNodeMap = [1:synNet.numNodes]; %randperm(synNet.numNodes); % mapping between class label and nodes
%synNet.classReAssignPercent = 0; % percentage of nodes to be quit from a community at each time step