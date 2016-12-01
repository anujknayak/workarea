% module to perform spectral clustering
function [c_0] = spectral_clustering(W, K)

% number of dominant singular values - consider moving this choice to the
% top level - parameterize this [TODO]
% number of dominant singular values
k = size(W, 1)/4;

% singular value decompositioins
[UMat, SigMat, VMat] = svd(W);
% Sigma Matrix - matrix of singular values
SigMat = SigMat(1:k, 1:k);
% U - left singular vector matrix
UMat = UMat(:, 1:k);
% V - right singular vector matrix
VMat = VMat(:, 1:k);
% input to k-means clustering
ZMat = [UMat*sqrt(SigMat),VMat*sqrt(SigMat)];

% k-means clustering
[c_0 C] = kmeans(ZMat, K);
