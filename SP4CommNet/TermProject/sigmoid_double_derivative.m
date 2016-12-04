% function to compute the double derivative of sigmoid function
function [y] = sigmoid_double_derivative(x)
y = -(exp(x)-1).*exp(x)./((exp(x)+1).^3);