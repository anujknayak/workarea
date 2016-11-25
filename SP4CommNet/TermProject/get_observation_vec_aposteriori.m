function [yVec, m_abVec, n_abVec] = get_observation_vec_aposteriori(W, classLabelList, numClasses)

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

m_abVec = reshape(m_abMat, numClasses^2, 1, []);
n_abVec = reshape(n_abMat, numClasses^2, 1, []);
yVec = reshape(yMat, [], 1);


