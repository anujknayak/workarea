function [y] = second_order_ekf_term_compute(psiVec, RMat)
doubleDerivPsi =  sigmoid_double_derivative(psiVec);
coreTerm = (doubleDerivPsi.*diag(RMat))*(doubleDerivPsi.*diag(RMat)).';
y = 3/4*coreTerm;
