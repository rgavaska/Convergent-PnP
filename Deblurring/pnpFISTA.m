function [x_curr,converged,iters,E,P,obj] = ...
                    pnpFISTA(x0,gradf,W,rho_D,I,tol,maxiters,objfun,runISTA)
%PNPFISTA: (Scaled) Plug-and-play FISTA
% Input arguments:
% x0 = Initial point
% gradf = Handle to gradient function of the data fidelity term
% W = Handle to denoiser
% rho_D = Product of step-size (rho) and scaling matrix (D), used in the
%         gradient descent step.
%         Note: For standard (F)ISTA, set rho to be the scalar step-size.
%         For scaled (F)ISTA, set rho_D to be the product of the scalar
%         step-size (rho) and the scaling matrix D.
% I = Ground-truth image to compare PSNR (optional), can be empty ([])
% tol = Tolerance (optional)
% maxiters = Max. no. of iterations (optional)
% objfun = Function to calculate objective value (optional)
% runISTA = Flag, set to true to run ISTA instead of FISTA (default is FISTA)
%
% Output arguments:
% x_curr = Output
% converged = Flag, true of the algorithm converged
% iters = No. of iterations executed
% E = Iteration-wise errors, norm(x_{k+1} - x_k)
% P = Iteration-wise PSNRs
% obj = Iteration-wise objective function values
% 

if(~exist('I','var') || isempty(I))
    calcPSNR = false;
else
    calcPSNR = true;
end
if(~exist('tol','var') || isempty(tol))
    tol = -1;   % Run for max. no. of iterations
end
if(~exist('maxiters','var') || isempty(maxiters))
    maxiters = 30;
end
if(~exist('objfun','var') || isempty(objfun))
    calcObj = false;
else
    calcObj = true;
end
if(~exist('runISTA','var') || isempty(runISTA))
    runISTA = false;
end

iters = 1;
x_prev = x0;
y_curr = x0;
t_curr = 1;
E = nan(1,maxiters);
if(calcPSNR)
    P = nan(1,maxiters);
    P(1) = psnr(x0,I,1);
end
if(calcObj)
    obj = nan(1,maxiters);
end
while(true)
    v_curr = y_curr - gradf(y_curr)./rho_D;
    x_curr = W(v_curr);
    if(runISTA)
        t_next = 1;
    else
        t_next = (1 + sqrt(1 + 4*t_curr*t_curr))/2;
    end
    y_next = x_curr + ((t_curr-1)/t_next)*(x_curr - x_prev);
    
    % Calculate PSNR, objective value etc.
    if(calcObj)
        obj(iters) = objfun(x_curr,v_curr);
    end
    if(calcPSNR)
        P(iters+1) = psnr(x_curr,I,1);
        fprintf('Iteration = %d,\tPSNR = %f\n',iters,P(iters+1));
    end
    err = euclNorm(x_curr - x_prev);
    E(iters) = err;
    if(err < tol)
        converged = true;
        E(iters+1:end) = [];
        break;
    end
    if(iters==maxiters)
        if(err < tol)
            converged = true;
        else
            converged = false;
        end
        break;
    end
    
    % Transition to next iteration
    iters = iters+1;
    y_curr = y_next;
    x_prev = x_curr;
    t_curr = t_next;
end

E(iters+1:end) = [];
if(calcPSNR)
    P(iters+1:end) = [];
else
    P = [];
end

end


function n = euclNorm(y)
% Euclidean norm of vector or matrix

n = sqrt(sum(y(:).*y(:)));

end
