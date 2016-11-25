M = 6;N = 8;
r = [0:1:min(N, M)];
d = (M-r).*(N-r);
figure(1);  plot(r, d, '-o', 'linewidth', 2);xlabel('r (multiplelxing gain)');ylabel('d (diversity gain)');title(['diversity multiplexing trade-off: ' num2str(M) 'X' num2str(N)]);
set(gca, 'fontsize', 20);grid on;
