function [pt] = posterior_prob_compute(m_abVec, n_abVec, psiEst_t, psiEstInit, REstInit)

% factor 1
pt_fact1 = sum(m_abVec.*log(sigmoid_fun(psiEst_t)) + (n_abVec - m_abVec).*log(1-sigmoid_fun(psiEst_t)));
% factor 2
pt_fact2 = -1/2*(psiEst_t-psiEstInit).'*inv(REstInit)*(psiEst_t-psiEstInit);
% posterior probability
pt = pt_fact1 + pt_fact2;