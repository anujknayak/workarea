function [yVec, m_abVec, n_abVec] = get_observation_vec_aposteriori(W, classLabelList, numClasses)

for indClassRow = 1:numClasses
    nodeIndicesRow = (classLabelList == indClassRow);
    for indClassCol = 1:numClasses
        nodeIndicesCol = (classLabelList == indClassCol);
        edgeMat = W(nodeIndicesRow, nodeIndicesCol);
        m_abMat(indClassRow, indClassCol) = sum(sum(edgeMat));
        if indClassRow == indClassCol
            n_abMat(indClassRow, indClassCol) = size(edgeMat, 1)*(size(edgeMat, 1)-1);
        else
            n_abMat(indClassRow, indClassCol) = size(edgeMat, 1)*size(edgeMat, 2);
        end
        yMat(indClassRow, indClassCol) = m_abMat(indClassRow, indClassCol)/n_abMat(indClassRow, indClassCol);
    end
end

m_abVec = reshape(m_abMat, [], 1);
n_abVec = reshape(n_abMat, [], 1);
yVec = reshape(yMat, [], 1);


