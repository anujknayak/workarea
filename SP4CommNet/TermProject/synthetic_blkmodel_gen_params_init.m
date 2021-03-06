function [synNet] = synthetic_blkmodel_gen_params_init()

synNet.muZero = [0.2580 0.0834]; % [same_block; different block]
synNet.gaussEvolCovParamVecInit = 0.04; % diagonal only - to generate scaled identity matrix
synNet.gaussEvolCovParamVec = [0.01 0.0025]; % Covariance matrix for Gaussian random walk model
synNet.numNodes = 128; % number of nodes in the network
synNet.numClasses = 4; % number of class labels
synNet.classSizes = synNet.numNodes/synNet.numClasses*ones(1,synNet.numClasses);
synNet.classMemVecInit = repelem([1:synNet.numClasses], synNet.classSizes).'; % different class sizes can also be given. However, the literature does not demand this.
synNet.classMemNodeMap = [1:synNet.numNodes]; %randperm(synNet.numNodes); % mapping between class label and nodes
synNet.classReAssignPercent = 10; % percentage of nodes to be quit from a community at each time step

% % % % % % %% >>>> FOR DEBUG - DIFFERENT TEST PARAMETERS HERE
% % Enron network parameters
% synNet.muZero = [0.2580 0.0834]; % [same_block; different block]
% synNet.gaussEvolCovParamVecInit = 0.04; % diagonal only - to generate scaled identity matrix
% synNet.gaussEvolCovParamVec = [0.01 0.0025]; % Covariance matrix for Gaussian random walk model
% synNet.numNodes = 184; % number of nodes in the network
% synNet.numClasses = 7; % number of class labels
% %synNet.classSizes = synNet.numNodes/synNet.numClasses*ones(1,synNet.numClasses);
% %synNet.classMemVecInit = repelem([1:synNet.numClasses], synNet.classSizes).'; % different class sizes can also be given. However, the literature does not demand this.
% %synNet.classMemNodeMap = [1:synNet.numNodes]; %randperm(synNet.numNodes); % mapping between class label and nodes
% %synNet.classReAssignPercent = 0; % percentage of nodes to be quit from a community at each time step

% %% Parameters as in PhD Thesis Xu
% synNet.muZero = [0.2580 0.0834]; % [same_block; different block]
% synNet.gaussEvolCovParamVecInit = 0.02^2; % diagonal only - to generate scaled identity matrix
% synNet.gaussEvolCovParamVec = [0.1^2 0.05^2]; % Covariance matrix for Gaussian random walk model
% synNet.numNodes = 128; % number of nodes in the network
% synNet.numClasses = 4; % number of class labels
% synNet.classSizes = synNet.numNodes/synNet.numClasses*ones(1,synNet.numClasses);
% synNet.classMemVecInit = repelem([1:synNet.numClasses], synNet.classSizes).'; % different class sizes can also be given. However, the literature does not demand this.
% synNet.classMemNodeMap = [1:synNet.numNodes]; %randperm(synNet.numNodes); % mapping between class label and nodes
% synNet.classReAssignPercent = 10; % percentage of nodes to be quit from a community at each time step
