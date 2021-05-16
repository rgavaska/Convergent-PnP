function g = eval_regularizer(x,v,D)
%EVAL_REGULARIZER 
% x = Input argument
% v = Image with the property x = Wv, where W is the denoiser
% D = Normalization coefficients of the denoiser

y = v - x;
Dx = D.*x;
g = sum(y(:).*Dx(:));
g = g/2;

end

