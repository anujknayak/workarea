% input: numSnapShots
%        synNet.muZero
%              .gaussEvolCovParamVec
%              .classMemInit
% output: WMat
%         classMemVec

function [W, psiVec_t, classLabelList, numClasses, dbg] = synthetic_blkmodel_gen(numSnapShots, synNet, dbg)

% % >>>> for debug begin <<<<
% numSnapShots = 1e3;
% [synNet] = synthetic_blkmodel_gen_params_init();
% % >>>> for debug end <<<<

muZero = synNet.muZero;
gaussEvolCovParamVec = synNet.gaussEvolCovParamVec;
numNodes = synNet.numNodes;
numClasses = synNet.numClasses;
classSizes = synNet.classSizes;
classMemVecInit = synNet.classMemVecInit;
classMemNodeMap = synNet.classMemNodeMap;
classReAssignPercent = synNet.classReAssignPercent;
vCovMtx0 = synNet.gaussEvolCovParamVecInit*eye(numClasses^2); % initialization only
thetaMat = synNet.muZero(2)*ones(numClasses);thetaMat([0:numClasses-1]*numClasses + [1:numClasses]) = 0;thetaMat = thetaMat + diag(ones(1,numClasses)*synNet.muZero(1));
psiMat = logit_fun(thetaMat);psiVec = reshape(psiMat, [], 1);
psiVec = psiVec + mvnrnd(zeros(numClasses^2, 1).', vCovMtx0, 1).'; % initialization
vCovMtx = proc_noise_cov_mtx_gen(numClasses, synNet.gaussEvolCovParamVec(1), synNet.gaussEvolCovParamVec(2));
c_t = synNet.classMemVecInit; % this is always sorted Ex: 1 .....1 2 ..... 2 3 ....3 ... so on
classListAll = [1:numClasses];
thetaArr = zeros(numClasses, numSnapShots);
%classSizeList = zeros(); % initialize for speed
W = zeros(numNodes, numNodes, numSnapShots);
W = zeros(numNodes, numNodes, numSnapShots);
classLabelTemplate = repelem([1:numClasses].', classSizes, 1);

% for debug
%vArr = [];

for indSnapShot = 1:numSnapShots
    % generate correlated Gaussian vector
    v = mvnrnd(zeros(numClasses^2, 1), vCovMtx, 1).';
    %vArr = [vArr v];
    
    % Gaussian Random Walk
    if indSnapShot == 1
        psiVec_t(:, 1) = psiVec + v;
    else
        psiVec_t(:, indSnapShot) = psiVec_t(:, indSnapShot-1) + v;
    end
    
    % 10% random class re-assignment
    % generating a random mask of nodes
    classReAssIndicVec = [ones(round(classReAssignPercent/100*numNodes), 1);zeros(numNodes - round(classReAssignPercent/100*numNodes), 1)];
    classReAssIndicVec = classReAssIndicVec(randperm(numNodes));
    % class re-assignment
    c_t = mod((c_t-1) + (round(rand(numNodes, 1)*(numClasses-2))+1).*(classReAssIndicVec), numClasses)+1;
    [c_tSorted, sortIndices] = sort(c_t, 'ascend');
    
    % get the classes that exist in the current snapshot
    [classExistList, classSizesIndic] = unique(c_tSorted);
    % 1 x numClasses vector with the class size assigned at the corresponding indices
    classSizes = zeros(numClasses, 1);
    % determine the class sizes for the existing class labels
    classSizes(classExistList) = diff([classSizesIndic;numNodes+1]);
    
    classLabelList(:, indSnapShot) = c_t;
    % vector to matrix reshaping of psi vector
    psiMatSorted = repelem(reshape(psiVec_t(:, indSnapShot), numClasses, numClasses), classSizes, classSizes);
    psiMat(sortIndices, sortIndices) = psiMatSorted;
    
    % obtain adjacency matrix from the updated edge probability matrix (psi matrix)
    thetaMat = (1./(1+exp(-psiMat)));
    W(:,:,indSnapShot) = double(rand(numNodes) <= thetaMat);
    % removing the self edges
    W(:,:,indSnapShot) = mod(W(:,:,indSnapShot)+diag(diag(W(:,:,indSnapShot))), 2); % This should be fed to apriori blockmodel
    
    %figure(1);imagesc(W(sortIndices,sortIndices,indSnapShot));drawnow();
    %figure(1);spy(W(:,:,indSnapShot));drawnow();
    %figure(2);imagesc(thetaMat)
    %plotAdjacencyMat(W(:,:,indSnapShot), c_t, indSnapShot);
    %pause(0.05);
end

%figure;plot(mean(psiVec_t.'));

%dbg.thetaArr = sigmoid_fun(-[psiVec psiVec_t]);
%figure;plot(psiVec_t.');

dbg.vCovMtx = vCovMtx;
dbg.vCovMtx0 = vCovMtx0;

