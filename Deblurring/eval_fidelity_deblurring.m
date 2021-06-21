function f = eval_fidelity_deblurring(x,A,b)
%EVAL_FIDELITY_DEBLURRING
% Data fidelity term for the deblurring application
% x = Input argument
% A = Handle to linear degradation function
% b = Observed image/signal

f = A(x) - b;
f = f(:)' * f(:);   % Squared l2 norm
f = f/2;

end

