function f = eval_fidelity_inpainting(x,A,b)
%EVAL_FIDELITY_INPAINTING
% Data fidelity term for the inpainting application
% x = Input argument
% A = Mask (inpainting)
% b = Observed image

f = x(A==1) - b(A==1);
f = f(:)' * f(:);   % Squared l2 norm
f = f/2;

end

