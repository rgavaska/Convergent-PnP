function out = prox_inpainting(lambda,D,A,y,x)
% Function to compute scaled proximal map of f/lambda (data fidelity) for
% inpainting
% lambda = Scaling parameter (in denominator)
% D = Matrix of normalizing coefficients, same size as image (Metric-inducing pos. def. matrix )
% A = Decimation matrix
% y = Observed image
% x = Input argument (image)

out = D.*x + y/lambda;
denom = D + A/lambda;
out = out./denom;

end
