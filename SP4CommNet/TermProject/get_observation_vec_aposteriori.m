% function to obtain observation vector/block density vector for all the
% snapshots
% Inputs: W - adjacency matrix 
%         classLabelList - class label map for all nodes
%         numClasses - number of classes
% Output: yVec - matrix of block densities for all the snapshots
%         m_abVec - matrix of number of existing edges in each block
%         n_abVec - matrix of total number of possible edges in each block
function [yVec, m_abVec, n_abVec] = get_observation_vec_aposteriori(W, classLabelList, numClasses)

numSnapShots = size(W, 3);
m_abMat = zeros(numClasses, numClasses, numSnapShots);
n_abMat = zeros(numClasses, numClasses, numSnapShots);
yMat = zeros(numClasses, numClasses, numSnapShots);

for indSnapShot = 1:numSnapShots
    for indClassRow = 1:numClasses
        nodeIndicesRow = (classLabelList(:, indSnapShot) == indClassRow);
        nodeIndicesRowLen = sum(nodeIndicesRow);
        for indClassCol = 1:numClasses
            nodeIndicesCol = (classLabelList(:, indSnapShot) == indClassCol);
            % There is no way to speed up the following line
            m_abMatCurrent = nnz(W(nodeIndicesRow, nodeIndicesCol, indSnapShot));
            % determine the number of existing edges in the block
            m_abMat(indClassRow, indClassCol, indSnapShot) = m_abMatCurrent;
            % determine the total number of possible edges in the block
            if indClassRow == indClassCol
               % ignoring self-edges
               n_abMatCurrent = nodeIndicesRowLen*(nodeIndicesRowLen-1);
            else
               % for different classes self edges don't exist
               n_abMatCurrent = nodeIndicesRowLen*sum(nodeIndicesCol);
            end
            n_abMat(indClassRow, indClassCol, indSnapShot) = n_abMatCurrent;
            % compute the block density
            yMatValCurrent = m_abMat(indClassRow, indClassCol, indSnapShot)/n_abMat(indClassRow, indClassCol, indSnapShot);
            if (isnan(yMatValCurrent) || (yMatValCurrent == -inf) || (yMatValCurrent == inf)) && indSnapShot >1
                yMatValCurrent = yMat(indClassRow, indClassCol, indSnapShot-1)*0.01;
            elseif (isnan(yMatValCurrent) || (yMatValCurrent == -inf) || (yMatValCurrent == inf)) && indSnapShot == 1
                yMatValCurrent = 0.5/size(W, 1)^2;
            end
            yMat(indClassRow, indClassCol, indSnapShot) = yMatValCurrent;
        end
    end
end

% Matrix to vector - flattening
m_abVec = reshape(m_abMat, numClasses^2, 1, []);
n_abVec = reshape(n_abMat, numClasses^2, 1, []);
yVec = reshape(yMat, [], 1);
