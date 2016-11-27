function [psiEst_tSSBM, c_tSSBM] = get_static_sbm(W, numClasses)

numSnapShots = size(W, 3);
psiEst_tSSBM = zeros(numClasses^2, numSnapShots);
c_tSSBM = zeros(size(W, 1), numSnapShots);

for indSnapShot = 1:numSnapShots
    % perform spectral clustering
    [classLabelList] = spectral_clustering(W(:, :, indSnapShot), numClasses);
    % get observations vector
    [yVec, ~, ~] = get_observation_vec_aposteriori(W(:, :, indSnapShot), classLabelList, numClasses);
    % get Psi vector - logit of block density vector
    psiEst_tSSBM(:, indSnapShot) = logit_fun(yVec);
    c_tSSBM(:, indSnapShot) = classLabelList;
end

