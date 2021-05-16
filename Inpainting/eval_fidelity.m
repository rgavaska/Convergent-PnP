function f = eval_fidelity(x,A,b)
%EVAL_FIDELITY
% x = Input argument
% A = Mask (inpainting)
% b = Observed image

f = x(A==1) - b(A==1);
f = f(:)' * f(:);   % Squared l2 norm
f = f/2;

end

