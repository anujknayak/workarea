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
            yMatValCurrent = m_abMat(indClassRow, indClassCol, indSnapShot)/n_abMat(indClassRow, indClassCol, indSnapShot);
            % imposing the constraint - there is at least one edge
            if m_abMat(indClassRow, indClassCol, indSnapShot) == 0
				% overriding by fixed value 1e-4
                yMatValCurrent = 1e-4;
            end
            yMatRaw(indClassRow, indClassCol, indSnapShot) = yMatValCurrent;
            % The following conditions are required in real network when
            % there are no edges in a block (between a pair of classes)
			% CHECK IF THE FOLLOWING CODE IS REALLY NEEDED - ELSE DISCARD IN THE NEXT COMMIT
            if (isnan(yMatValCurrent) || (yMatValCurrent == -inf) || (yMatValCurrent == inf) || (yMatValCurrent == 0)) && indSnapShot >1
                yMatValCurrent = yMat(indClassRow, indClassCol, indSnapShot-1)*0.01;
            elseif (isnan(yMatValCurrent) || (yMatValCurrent == -inf) || (yMatValCurrent == inf) || (yMatValCurrent == 0)) && indSnapShot == 1
                yMatValCurrent = 0.5/size(W, 1)^2;
            end
            yMat(indClassRow, indClassCol, indSnapShot) = yMatValCurrent;
        end
    end
end

m_abVec = reshape(m_abMat, numClasses^2, []);
n_abVec = reshape(n_abMat, numClasses^2, []);
yVec = reshape(yMat, numClasses^2, []);
yVecRaw = reshape(yMatRaw, numClasses^2, []);


