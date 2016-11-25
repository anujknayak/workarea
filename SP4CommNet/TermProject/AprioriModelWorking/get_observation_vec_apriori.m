%
% inputs: W : adjacency matrix for all snapshots in time
%         classLabelList : class label vector
%         classSizesList : class size list corresponding to the labels in classLabelList
% outputs: yVec : observation vector
%
function [yVec, m_abVec, n_abVec] = get_observation_vec_apriori(W, classSizeList)

% % >>>> FOR DEBUG BEGIN <<<<
% % initialize the parameters for time evolving dynamic SBM
% [synNet] = synthetic_blkmodel_gen_params_init();
% 
% dbg = [];
% 
% % get the adjacency matrix and class labels for SBM
% [W, classLabelList, classSizeList, classMemNodeMap, dbg] = synthetic_blkmodel_gen(100, synNet, dbg);
% % >>>> FOR DEBUG END <<<<

numSnapShots = size(W, 3);
numClasses = size(classSizeList, 1);
for indSnapShot = 1:numSnapShots % loop for all time snap-shots
    for indClassRow = 1:numClasses % loop for all class labels - ROWS
        for indClassCol = 1:numClasses % loop for all class labels - COLUMNS
            % the following bifurcation is done to avoid minimum index error
            % get the row shift for the block under consideration
            if indClassRow == 1
                rowShift = 0;
            else
                rowShift =  sum(classSizeList(1:indClassRow-1, indSnapShot));   
            end
            
            % get the column shift for the block under consideration
            if indClassCol == 1
                colShift = 0;
            else
                colShift = sum(classSizeList(1:indClassCol-1, indSnapShot));
            end
            
            % Compute edge probabilities for each block
            if indClassRow == indClassCol
                yMat(indClassRow, indClassCol, indSnapShot) = sum(sum(W([1:classSizeList(indClassRow, indSnapShot)]+rowShift, [1:classSizeList(indClassCol, indSnapShot)]+colShift)))...
                /(classSizeList(indClassRow, indSnapShot)*(classSizeList(indClassCol, indSnapShot)-1));
                m_abMat(indClassRow, indClassCol, indSnapShot) = sum(sum(W([1:classSizeList(indClassRow, indSnapShot)]+rowShift, [1:classSizeList(indClassCol, indSnapShot)]+colShift)));
                n_abMat(indClassRow, indClassCol, indSnapShot) = (classSizeList(indClassRow, indSnapShot)*(classSizeList(indClassCol, indSnapShot)-1));
            else
                yMat(indClassRow, indClassCol, indSnapShot) = sum(sum(W([1:classSizeList(indClassRow, indSnapShot)]+rowShift, [1:classSizeList(indClassCol, indSnapShot)]+colShift)))...
                /(classSizeList(indClassRow, indSnapShot)*classSizeList(indClassCol, indSnapShot));
                m_abMat(indClassRow, indClassCol, indSnapShot) = sum(sum(W([1:classSizeList(indClassRow, indSnapShot)]+rowShift, [1:classSizeList(indClassCol, indSnapShot)]+colShift)));
                n_abMat(indClassRow, indClassCol, indSnapShot) = (classSizeList(indClassRow, indSnapShot)*(classSizeList(indClassCol, indSnapShot)-1));
            end
        end
    end
end

% Reshaping matrix to vector
yVec = reshape(yMat, size(yMat, 1) * size(yMat, 2), size(yMat, 3));
yVec = sigmoid_fun(yVec);

m_abVec = reshape(m_abMat, size(m_abMat, 1) * size(m_abMat, 2), size(m_abMat, 3));
n_abVec = reshape(n_abMat, size(n_abMat, 1) * size(n_abMat, 2), size(n_abMat, 3));
