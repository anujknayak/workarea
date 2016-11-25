% assignment
% H = [1 1 -1 1; ...
%     1 1 -1 -1;
%     1 1 1 1;
%     1 1 1 -1];

H = [1 1 1 -1;
    1 1 -1 1;
    1 -1 1 1;
    1 -1 -1 -1];

B = 10e6;

rhodB = 10;
rho = 10^(rhodB/10);

% % for debug begin
% H = [.1 .3 .7;...
%     .5 .4 .1; ...
%     .2 .6 .8];
% 
% rho = 10;
% B = 1;
% % for debug end

[U, sig, V] = svd(H);

gammaVec = (diag(sig)).^2.*rho;
conditionFlag = 0;

% total power constraint instead of average power constraint
while conditionFlag == 0
    gamma0 = size(gammaVec, 1)/(1+sum(1./gammaVec));
    if gamma0 < gammaVec(end)
        conditionFlag = 1;
    else
        gammaVec(end) = [];
    end
end

gammaVecZeroPadded = [gammaVec; zeros(size(H,2) - size(gammaVec, 1), 1)];

Pi_PVec = (1/gamma0 - 1./gammaVecZeroPadded);
Pi_PVec(find(abs(Pi_PVec) == inf)) = 0;
P_SigmaSqVec = Pi_PVec*rho;
P_SigmaSqVec(find(abs(P_SigmaSqVec) == inf)) = 0;

disp(' ');

disp('sigma_iSq');
disp(diag(sig).');

disp('P_i/P');
disp(Pi_PVec.');

disp('P_i/sigmaSq');
disp(P_SigmaSqVec.');

disp('gamma0');
disp(gamma0.');

disp('gamma_i');
disp(gammaVec.');

COptimal = B*log2(gammaVec/gamma0);
disp('Capacity of each spatial channel');
disp(COptimal.');

disp('Capacity with optimal power allocation = ');
COptimalSum = sum(COptimal);
disp(COptimalSum);

% disp('Capacity with unknown CSI at the Transmitter');
% CUnknown = B*log2(det(eye(size(H, 1))+rho/size(H, 2)*H*H.'));
% disp(CUnknown);



