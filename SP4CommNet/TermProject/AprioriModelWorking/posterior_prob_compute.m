function [pt] = posterior_prob_compute(m_abVec, n_abVec, psiEst_t, psiEstInit, REstInit)

% factor 1
pt_fact1 = sum(m_abVec(:,1).*log(sigmoid_fun(psiEst_t(:,1))) + (n_abVec(:,1) - m_abVec(:,1)).*log(1-sigmoid_fun(psiEst_t(:,1))));
% factor 2
pt_fact2 = -1/2*(psiEst_t-psiEstInit).'*inv(REstInit)*(psiEst_t-psiEstInit);
% posterior probability
pt = pt_fact1 + pt_fact2;