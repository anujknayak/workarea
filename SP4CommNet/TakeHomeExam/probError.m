clear all;close all;clc;
set(groot,'defaultLineLineWidth',2);
A1dB = [0:1:15];
A1 = 10.^(A1dB/20);
A2 = A1;
%A2 = 0*ones(size(A1));
rho = 0.75;
pbOptimLB = max(qfun(A1), .5*qfun(sqrt(A1.^2+A2.^2-2*A1.*A2*rho)));
pbOptimUB = qfun(A1) + .5*qfun(sqrt(A1.^2+A2.^2-2*A1.*A2*rho));
pbSingleUserMF = .5*qfun(A1-A2*rho)+.5*qfun(A1+A2*rho);
pbDecorr = qfun(A1*sqrt((1-rho^2)));
pbSicDecorr = qfun(A1).*(1-qfun(A2*(sqrt(1-rho^2)))) + qfun(A2*(sqrt(1-rho^2))).*0.5.*(qfun(A1+2*A2*rho)+qfun(A1-2*A2*rho));
% for ind = 1:length(A1)
% A(1,1) = A1(ind);A(2,2) = A2(ind);
% R = [1 rho;rho 1]+inv(A).^2;
% M = inv(R);
% MR = M*R;MR1 = MR(1,1);
% MRM = M*R*M;MRM1 = MRM(1,1);
% muVal = A1(ind).*MR1./sqrt(MRM1);
% B2 = A2(ind)*(MR(1,2));beta2 = B2/(A1(ind)*MR1);
% lambdaSqVal = muVal.^2*beta2^2;
% phMMSE(ind) = qfun(muVal/sqrt(1+lambdaSqVal));
% phMMSEExact(ind) = .5*qfun(A1(ind)*MR1/sqrt(MRM1)*(1+beta2)) + .5*qfun(A1(ind)*MR1/sqrt(MRM1)*(1-beta2));
% end

figure;semilogy(A1dB, pbOptimLB, 'b-o');hold on;semilogy(A1dB, pbOptimUB, 'r-^');semilogy(A1dB, pbSingleUserMF, 'k-x');semilogy(A1dB, pbDecorr, 'm-s');semilogy(A1dB, pbSicDecorr, 'k-s');
%semilogy(A1dB, phMMSEExact, 'r-x');
grid on;xlabel('SNR (dB)');ylabel('BER');xlim([min(A1dB) max(A1dB)]);
title(['$\rho$ = ' num2str(rho) ', $\frac{A2}{A1}$ = ' num2str(A2(1)/A1(1))], 'interpreter', 'latex')
set(gca, 'fontsize', 18);legend({'optimum lower bound','optimum upper bound','single user matched filter','decorrelator', 'SIC - decor 1st stage'});
%set(gca, 'fontsize', 18);legend({'optimum lower bound','optimum upper bound','single user matched filter','decorrelator','LMMSE Exact'});