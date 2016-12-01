%
% inputs: W : adjacency matrix for all snapshots in time
%         classLabelList : class label vector
%         classSizesList : class size list corresponding to the labels in classLabelList
% outputs: yVec : observation vector
%
function [yVec, m_abVec, n_abVec] = get_observation_vec_apriori(W, classLabelList, numClasses)

% % >>>> FOR DEBUG BEGIN <<<<
% % initialize the parameters for time evolving dynamic SBM
% [synNet] = synthetic_blkmodel_gen_params_init();
% 
% dbg = [];
% 
% % get the adjacency matrix and class labels for SBM
% [W, psiVec_t, classLabelList, numClasses, dbg] = synthetic_blkmodel_gen(100, synNet, dbg);
% % >>>> FOR DEBUG END <<<<

numSnapShots = size(W, 3);

for indSnapShot = 1:numSnapShots
    for indClassRow = 1:numClasses
        nodeIndicesRow = (classLabelList(:, indSnapShot) == indClassRow);
        for indClassCol = 1:numClasses
            nodeIndicesCol = (classLabelList(:, indSnapShot) == indClassCol);
            edgeMat = W(nodeIndicesRow, nodeIndicesCol, indSnapShot);
            m_abMat(indClassRow, indClassCol, indSnapShot) = sum(sum(edgeMat));
            if indClassRow == indClassCol
                n_abMat(indClassRow, indClassCol, indSnapShot) = size(edgeMat, 1)*(size(edgeMat, 1)-1);
            else
                n_abMat(indClassRow, indClassCol, indSnapShot) = size(edgeMat, 1)*size(edgeMat, 2);
            end
            yMat(indClassRow, indClassCol, indSnapShot) = m_abMat(indClassRow, indClassCol, indSnapShot)/n_abMat(indClassRow, indClassCol, indSnapShot);
        end
    end
end

m_abVec = reshape(m_abMat, numClasses^2, []);
n_abVec = reshape(n_abMat, numClasses^2, []);
yVec = reshape(yMat, numClasses^2, []);
yVec(isnan(yVec) | (yVec == inf)) = 0.5/numClasses^2;
yVec(yVec == 0) = 0.5/numClasses^2;


