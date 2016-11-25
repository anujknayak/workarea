
set(groot,'defaultLineLineWidth',2);

A2byA1 = [0:.1:2];
rho = .75;
ameOptim = min(1, 1+A2byA1.^2-2*rho*A2byA1);
ameSingleUserMF = (max(0, 1-A2byA1*rho)).^2;
ameDecorr = ones(size(A2byA1))*(1-rho.^2);
ameSicDecorr = min(ones(size(A2byA1)), (A2byA1.^2)*(1-rho^2)+(max(zeros(size(A2byA1)), 1- 2*A2byA1*rho)).^2);

figure;plot(A2byA1, ameOptim, 'b-x');hold on;plot(A2byA1, ameSingleUserMF, 'k-^');plot(A2byA1, ameDecorr, 'm-s');plot(A2byA1, ameSicDecorr, 'r-o')
grid on;xlabel('A_2/A_1');ylabel('BER');
set(gca, 'fontsize', 18);legend({'optimum','single user matched filter','decorrelator','SIC - decor 1st stage'});