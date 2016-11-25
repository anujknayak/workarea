%%  Goldsmith 14.7 - b - rho = 0.01

% link gains
g1 = 1;
g2 = 3;
g3 = 5;
% cross correlation value
rho = 0.01;
% gain matrix
gainMtx = [  g1       g2*rho    g3*rho; ...
           g1*rho     g2        g3*rho; ...
           g1*rho   g2*rho        g3];

% target SINR values
gamma = [10;10;10];
% noise power
noisePwr = [1;1;1];

DMtx = diag(gamma);
FMtx = [gainMtx - diag(diag(gainMtx))].*((ones(size(gainMtx, 2), 1)./diag(gainMtx))*ones(1,size(gainMtx, 2)));
uMtx = gamma.*noisePwr./diag(gainMtx);

% computing powers of individual users to satisfy the SINR requirement of each user
p = inv(eye(size(gainMtx, 1)) - DMtx*FMtx)*uMtx;
disp('p = ');
disp(p);
disp('Perron-Frobenius Eigen Value');
disp(max(abs(eig(DMtx*FMtx))));
