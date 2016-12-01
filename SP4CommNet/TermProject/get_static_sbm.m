function [psiEst_tSSBM, c_tSSBM] = get_static_sbm(W, numClasses)

numSnapShots = size(W, 3);
psiEst_tSSBM = zeros(numClasses^2, numSnapShots);
c_tSSBM = zeros(size(W, 1), numSnapShots);

for indSnapShot = 1:numSnapShots
    % perform spectral clustering
    [classLabelList] = spectral_clustering(W(:, :, indSnapShot), numClasses);
    % get observations vector
    [yVec, ~, ~] = get_observation_vec_apriori(W(:, :, indSnapShot), classLabelList, numClasses);
    % THE FOLLOWING CHUNK OF CODE (DEBUG BEGIN to END) IS A HACK
    % >>>> FOR DEBUG - BEGIN
    yVec(yVec >= 1) = 0.9999;
    yVec(yVec <= 0) = 0.0001;
    % >>>> FOR DEBUG END
    % get Psi vector - logit of block density vector
    psiEst_tSSBM(:, indSnapShot) = logit_fun(yVec);
    yVecAll(:, indSnapShot) = yVec;
    c_tSSBM(:, indSnapShot) = classLabelList;
end
