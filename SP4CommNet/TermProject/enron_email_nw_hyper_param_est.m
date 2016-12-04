% function for hyperparameter estimation for Enron email network
% Apriori blockmodel only

sdiagList = kron(10.^[-2:0], [2:4:10]);
snbList = kron(10.^[-3:-1], [2:4:10]);

numIters = 1; % number of iterations
burnInPeriod = 50; % in terms of number of snapshots
numSnapShots = 120;

% enable for [apriori blockmodel, aposteriori blockmodel, static stochastic blockmodel] - respectively
blockEnableVec = [1 0 0];

% initialize the parameters for time evolving dynamic SBM
[synNet] = enron_email_nw_synthetic_blkmodel_gen_params_init();
Params = enron_email_nw_blkmodel_params_init([]);

mseAprioriVec = zeros(1, numIters);
[W, psiVec_t, yVec_t, classLabelList, numClasses] = enron_email_nw_get_adjacency_mat();


for indsdiag = 1:length(sdiagList)
    for indsnb = 1:length(snbList)
        if sdiagList(indsdiag) < snbList(indsnb)
            continue;
        end
        disp([sdiagList(indsdiag) snbList(indsnb)]);
        dbg = [];
        GammaMat = proc_noise_cov_mtx_gen(synNet.numClasses, sdiagList(indsdiag), snbList(indsnb));
        ParamsApriori = Params.apriori;
        ParamsApriori.GammaMat = GammaMat;
        
        % >>>> FOR DEBUG BEGIN
        %ParamsApriori.GammaMat = rand(size(GammaMat));
        % >>>> FOR DEBUG END
        
        % get the apriori SBM
        [psiEst_tApriori, yVecApriori, ~, ~, ~, psiEst_tMinus1Apriori] = get_apriori_blkmodel(W, classLabelList, numClasses, ParamsApriori);
        
        % computing tracking error for apriori SBM - this should be straight forward
        % compute error
        errAprioriMat = sigmoid_fun(psiEst_tMinus1Apriori)-yVec_t;
        % discard the elements corresponding to the burn-in period
        errAprioriMat(:, 1:burnInPeriod) = 0;
        % compute MSE
        mseApriori = sum(sum(abs(errAprioriMat).^2))/((numSnapShots-burnInPeriod)*(synNet.numNodes^2));
        % collect MSEs in a vector
        mseAprioriVec(indsdiag, indsnb) = mseApriori;
        week89Vec = zeros(1,numSnapShots);week89Vec(89) = 1;
        figure(1);plot((yVec_t(16, :).'));hold on;plot(sigmoid_fun(psiEst_tApriori(16,:)).','linewidth', 2);plot(week89Vec, 'k');hold off;drawnow();pause;
    end
end

%figure;surf(snbList, sdiagList, mseAprioriVec);xlabel('sdiag');ylabel('snb');set(gca, 'XScale', 'log');set(gca, 'YScale', 'log');set(gca, 'ZScale', 'log');
figure;plot(sdiagList,mseAprioriVec);xlabel('sdiag');ylabel('snb');set(gca, 'XScale', 'log');set(gca, 'YScale', 'log');