% case ideal
H = [.1 .3 .4; .3 .2 .2; .1 .3 .7];
[U, sigH, V] = svd(H);
sigH = diag(sigH);
disp(' ');
disp(['Singular values of H = [' num2str(sigH.') ']']);

HEstA =[0.095 0.283 0.382;
 0.281 0.182 0.193;
 0.09 0.273 0.625];

[U, sigA, V] = svd(HEstA);
sigA = diag(sigA);
disp(['Singular values of HEstA = [' num2str(sigA.') ']']);

% case b
HEstB =[0.110 0.330 0.437;
 0.324 0.206 0.222;
 0.103 0.316 0.789];

[U, sigB, V] = svd(HEstB);
sigB = diag(sigB);
disp(['Singular values of HEstB = [' num2str(sigB.') ']']);
disp(' ');

% computing condition numbers
fprintf('Condition number for H = %0.2f\n\n', cond(H));

% absolute deviation
sigDevAbsA = abs(sigA-sigH);
sigDevAbsB = abs(sigB-sigH);
disp(['Absolute deviation in singular values of HEstA = [' num2str(sigDevAbsA.') ']']);
disp(['Absolute deviation in singular values of HEstB = [' num2str(sigDevAbsB.') ']']);

% relative deviation
sigDevRelA = abs(sigA-sigH)./abs(sigH);
sigDevRelB = abs(sigB-sigH)./abs(sigH);
disp(['Relative deviation in singular values of HestA = [' num2str(sigDevRelA.') ']']);
disp(['Relative deviation in singular values of HestA = [' num2str(sigDevRelB.') ']']);



