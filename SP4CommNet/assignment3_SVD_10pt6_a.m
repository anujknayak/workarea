% case ideal
H = [.1 .3 .4; .3 .2 .2; .1 .3 .7];
[U, sigH, V] = svd(H);
sigH = diag(sigH);
disp(' ');
disp(['Singular values of H = [' num2str(sigH.') ']']);

% Identity Matrix
eyeMat = eye(3);

[U, sigEye, V] = svd(eyeMat);
sigEye = diag(sigEye);
disp(['Singular values of identity matrix = [' num2str(sigEye.') ']']);

% computing condition numbers
fprintf('Condition number for H = %0.2f\n', cond(H));
fprintf('Condition number for identity Matrix = %0.2f\n', cond(eyeMat));

numIters = 1000;

pfNormH = sqrt(sum(sum(abs(H.^2))));
pfNormEye = sqrt(sum(sum(abs(eyeMat.^2))));

snrList = 10.^[1:5];
singValVecH = zeros(1, min(size(H)));
singValVecEye = zeros(1, min(size(eyeMat)));

for snrInd = 1:length(snrList)
    for iterInd = 1:numIters
        % add perturbation
        HPerturbed = H + randn(size(H))*pfNormH/sqrt(snrList(snrInd));
        eyeMatPerturbed = eyeMat + randn(size(eyeMat))*pfNormEye/sqrt(snrList(snrInd));
        [U, sigHPerturbed, V] = svd(HPerturbed);
        [U, sigEyePerturbed, V] = svd(eyeMatPerturbed);
        sigHPerturbed = diag(sigHPerturbed);
        sigEyePerturbed = diag(sigEyePerturbed);
        % normalized error matrix
        errMatH(snrInd, :, iterInd) = permute((sigHPerturbed - sigH)./abs(sigH), [2 3 1]);
        errMatEye(snrInd, :, iterInd) = permute((sigEyePerturbed - sigEye)./abs(sigEye), [2 3 1]);
        if snrInd == 1 && iterInd == 1
            disp(['Singular values of H with 10dB perturbation = [' num2str(sigHPerturbed.') ']']);
            disp(['Singular values of identity matrix with 10dB perturbation = [' num2str(sigEyePerturbed.') ']']);
        end
    end
end

varMatH = sum(abs(errMatH.^2), 3)/numIters;
varEye = sum(abs(errMatEye.^2), 3)/numIters;

figure(1);
semilogy(10*log10(snrList), varMatH(:,end), '-o', 'linewidth', 2);hold on;
semilogy(10*log10(snrList), varEye(:,end), '-^', 'linewidth', 2);grid on;
xlabel('$$\frac{\Sigma |h_{ij}|^2}{\sigma_n^2}$$ (in dB)', 'Interpreter','latex');ylabel('var($$\frac{\tilde{\sigma_{K}}-\sigma_{K}}{\sigma_K}$$)','Interpreter','latex');
legend('H','Identity Matrix');set(gca, 'fontsize', 15);title('Error variance of smallest singular value (normalized)');