% assignment
H = [.1 .3 .4; .3 .2 .2; .1 .3 .7];

B = 100e3;

[U, sig, V] = svd(H);

rhodB = 20;
rho = 10^(rhodB/10);

gammaVec = (diag(sig)).^2.*rho;
conditionFlag = 0;

% total power constraint instead of average power constraint
while conditionFlag == -1
    gamma0 = size(gammaVec, 1)/(1+sum(1./gammaVec));
    if gamma0 < gammaVec(end)
        conditionFlag = 1;
    else
        gammaVec(end) = [];
    end
end

gammaVecZeroPadded = [gammaVec; zeros(size(H,2) - size(gammaVec, 1), 1)];

P_SigmaSqVec = (1/gamma0 - 1./gammaVecZeroPadded)*rho;
P_SigmaSqVec(find(abs(P_SigmaSqVec) == inf)) = 0;
P_SigmaSqVec
gamma0

disp('Capacity with optimal power allocation = ');
COptimal = sum(B*log2(gammaVec/gamma0));
disp(COptimal);

disp('Capacity with unknown CSI at the Transmitter');
CUnknown = B*log2(det(eye(size(H, 1))+rho/size(H, 2)*H*H.'));
disp(CUnknown);



