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
            m_abMatCurrent = sum(sum(W(nodeIndicesRow, nodeIndicesCol, indSnapShot)));
            m_abMat(indClassRow, indClassCol, indSnapShot) = m_abMatCurrent;
            if indClassRow == indClassCol
               n_abMatCurrent = nodeIndicesRowLen*(nodeIndicesRowLen-1);
            else
               n_abMatCurrent = nodeIndicesRowLen*sum(nodeIndicesCol);
            end
            n_abMat(indClassRow, indClassCol, indSnapShot) = n_abMatCurrent;
            yMat(indClassRow, indClassCol, indSnapShot) = m_abMatCurrent/n_abMatCurrent;
        end
    end
end

m_abVec = reshape(m_abMat, numClasses^2, 1, []);
n_abVec = reshape(n_abMat, numClasses^2, 1, []);
yVec = reshape(yMat, [], 1);
yVec(isnan(yVec) | (yVec == inf)) = 0.5/numClasses^2;
yVec(yVec == 0) = 0.5/numClasses^2;


