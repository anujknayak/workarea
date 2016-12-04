function [eigMax1, eigMax2] = eigen_val_compute(secOrEkfMat, sigMat)

[~, sigLhs, ~] = svd(secOrEkfMat);
eigMax1 = max(diag(sigLhs));

[~, sigRhs, ~] = svd(sigMat);
eigMax2 = max(diag(sigRhs));