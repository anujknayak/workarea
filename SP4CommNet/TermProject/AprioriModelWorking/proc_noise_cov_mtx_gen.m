function [vCovMtx] = proc_noise_cov_mtx_gen(K, sdiag, snb)

% >>>> FOR DEBUG BEGIN <<<<
%K = 4;
%sdiag = 0.01;
%snb = .0025;
% >>>> FOR DEBUG END <<<<

blkPairVec1 = repmat([1:K], 1, K);
blkPairVec2 = kron([1:K], ones(1,K));

vCovMtx = zeros(K^2);

for rowInd = 1:K^2
    for colInd = 1:K^2
        if rowInd == colInd
            vCovMtx(rowInd, colInd) = sdiag;
        else
            trueCond = sum(ismember([blkPairVec1(rowInd) blkPairVec2(rowInd)], [blkPairVec1(colInd) blkPairVec2(colInd)]));
            if trueCond
                vCovMtx(rowInd, colInd) = snb;
            else
                vCovMtx(rowInd, colInd) = 0;
            end
        end
    end
end

%figure(1);imagesc(vCovMtx);
%brkpnt1 = 1;

    