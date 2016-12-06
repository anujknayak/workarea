function [psiEst_tAll, c_tHatAll, yVecAll] = get_aposteriori_blkmodel(W, numClasses, Params)

% % >>>> FOR DEBUG BEGIN <<<<
% numSnapShots = 200;
% [synNet] = synthetic_blkmodel_gen_params_init();
% dbg = [];
% [W, psiVec_t, classLabelList, numClasses, dbg] = synthetic_blkmodel_gen(numSnapShots, synNet, dbg);
% Params.GammaMat = dbg.GammaMat;
% Params.GammaMat0 = dbg.GammaMat0;
% Params.numClasses = synNet.numClasses;
% Params.muZero = synNet.muZero;
% % >>>> FOR DEBUG END <<<<

maxIter = 30;
numSnapShots = size(W, 3);
numNodes = size(W, 1);

% initialize class membership - spectral clustering
[c_0] = spectral_clustering(W(:,:,1), numClasses);
% compute block densities Y^t
[yVec, m_abVec, n_abVec] = get_observation_vec_aposteriori(W(:,:,1), c_0, numClasses);

% compute EKF equations
FMat = eye(numClasses^2);
ekfParams.GammaMat = Params.GammaMat; % consider estimating this hyperparameter

% computing init hyperparameters
% estimating muZero
muZeroVec = sigmoid_fun(yVec);
psiEst_t = muZeroVec;
% estimating Gamma0
sig_abSq = yVec.*(1-yVec)./n_abVec;SigMat = diag(sig_abSq);
GMat = diag(1./yVec + 1./(1-yVec));
GammaMatZeroInit = GMat*SigMat*GMat.';
REst_t = FMat*GammaMatZeroInit*FMat.' + ekfParams.GammaMat;

% initialization for local search (hill climbing) algorithm
c_tHat = c_0;
psiEst_tAll = zeros(numClasses^2, numSnapShots);
c_tHatAll = zeros(numNodes, numSnapShots);
yVecAll = zeros(numClasses^2, numSnapShots);

% enumerating class membership vector copies - to avoid confusion
% c_tTilda - class membership trial for each iteration
% c_tBar - class membership updated for each hypothesis (node-class membership) when positive gradient (uphill) is encountered
% c_tHat - the best class membership after current iteration - it is also the best final class membership vector

for indSnapShot = 1:numSnapShots
    
    fprintf('snapshot %d\n',indSnapShot);
    [yVec, m_abVec, n_abVec] = get_observation_vec_aposteriori(W(:,:,indSnapShot), c_tHat, numClasses);
    yVecBar = yVec;
    % Sigma matrix - observation noise variance computation
    sig_abSq = yVec.*(1-yVec)./n_abVec;
    SigMat = diag(sig_abSq);
    % Extended Kalman Filter
    [psiEst_t, REst_t, psiEst_tMinus1, REst_tMinus1] = extended_kalman_filter(yVec, psiEst_t, FMat, REst_t, ekfParams.GammaMat, SigMat); % n_abVec);
    % compute log posterior
    [pt] = posterior_prob_compute(m_abVec, n_abVec, psiEst_t, psiEst_tMinus1, REst_tMinus1);
    
    iterLoop = 0;
    % Local search algorithm
    % This determines how many steps are climbed uphill
    while iterLoop < maxIter
        ptBar = -inf;
        c_tTilda = c_tHat;
        %ptArr = [];
        for indNode = 1:numNodes
            for indClass = 1:numClasses
                if c_tTilda(indNode) == indClass
                    continue;
                else
                    c_tTilda(indNode) = indClass;
                    % compute block densities
                    [yVec, m_abVec, n_abVec] = get_observation_vec_aposteriori(W(:,:,indSnapShot), c_tTilda, numClasses);
                    % EKF equations
					% Sigma matrix - observation noise variance computation
                    sig_abSq = yVec.*(1-yVec)./n_abVec;
                    SigMat = diag(sig_abSq);
                    % Extended Kalman Filter
                    [psiEst_tTilda, REst_tTilda, psiEst_tMinus1, REst_tMinus1] = extended_kalman_filter(yVec, psiEst_t, FMat, REst_t, ekfParams.GammaMat, SigMat); %n_abVec);
                    % compute posterior probability
                    [ptTilda] = posterior_prob_compute(m_abVec, n_abVec, psiEst_tTilda, psiEst_tMinus1, REst_tMinus1);
                    %ptArr = [ptArr ptTilda];
                    %figure(1);plot(ptArr, 'b-o');drawnow();
                    if ptTilda > ptBar % Visited solution is better than the best neighboring solution
                        ptBar = ptTilda;
                        psiEst_tBar = psiEst_tTilda;
                        c_tBar = c_tTilda;
                        REst_tBar = REst_t;
                        yVecBar = yVec;
                    end
                    % Reset the class membership of the current node
                    c_tTilda(indNode) = c_tHat(indNode);
                end
            end
        end
        if ptBar > pt % Best neighboring solution is better than the current best solution
            pt = ptBar;
            psiEst_t = psiEst_tBar;
            c_tHat = c_tBar;
        else % Reached local maximum
            disp(iterLoop);
            break;
        end
        iterLoop = iterLoop + 1;
    end
    psiEst_tAll(:, indSnapShot) = psiEst_t;
    c_tHatAll(:, indSnapShot) = c_tHat;
    yVecAll(:, indSnapShot) = yVecBar;
%    figure(1);imagesc(W(:,:,indSnapShot));
%    figure(2);plot(c_tHat, 'r-o');hold on;plot(c_0);hold off;drawnow();
end

%figure(1);plot(sigmoid_fun(psiEst_tAll.'), 'linewidth', 2);hold on;plot(yVecAll.');figure(2);plot(yVecAll.'-sigmoid_fun(psiEst_tAll.'));





