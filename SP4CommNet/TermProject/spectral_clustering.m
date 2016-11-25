function [c_0] = spectral_clustering(W, K)

k = size(W, 1)/4;

[UMat, SigMat, VMat] = svd(W);
SigMat = SigMat(1:k, 1:k);
UMat = UMat(:, 1:k);
VMat = VMat(:, 1:k);
ZMat = [UMat*sqrt(SigMat),VMat*sqrt(SigMat)];

[c_0 C] = kmeans(ZMat, K);