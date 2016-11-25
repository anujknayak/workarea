M = 7;
phiVec = [-pi/2:pi*1e-3:pi/2];
theta = 0;
dOverLambda = .5;
B = 1/M*abs(sum(exp(-sqrt(-1)*2*pi*([0:1:M-1].')*ones(1,length(phiVec))*dOverLambda.*(ones(M, 1)*(sin(theta) - sin(phiVec))))));
figure;plot(phiVec, B, 'linewidth', 2);xlabel('\phi');title(['d = ' num2str(dOverLambda) '\lambda, M = ' num2str(M)]);xlim([min(phiVec) max(phiVec)]);
set(gca, 'fontsize', 20);grid on;
