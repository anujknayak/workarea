rho12 = [0:.01:.5];rho12Len = length(rho12);
rho23 = [0:.01:0.5];rho23Len = length(rho23);
rho12Grid = rho12.'*ones(1, rho23Len);
rho23Grid = ones(rho12Len, 1)*rho23;
x1 = .5*rho12Grid./((1-rho12Grid).*(1-rho23Grid));
x2 = (rho12Grid.*(3+rho23Grid-2*rho23Grid.^2))./(2-2*rho23Grid.^2);
x3 = ((1-rho12Grid).^2) + (2*rho12Grid.*(1+rho23Grid))./(1-rho23Grid.^2);

figure(1);surf(rho23, rho12, x1);
figure(2);surf(rho23, rho12, x2);
figure(3);surf(rho23, rho23, x3);
figure(4);surf(rho23, rho12, x1-x3);