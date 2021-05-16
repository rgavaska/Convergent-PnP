function [x_curr,converged,iters,psnr_vals,...
          primal_residue,dual_residue,obj] = ...
                    pnpADMM(z0,u0,rho,prox_f,W,I,tol,maxiters,objfun)
%PNPADMM: (Scaled) Plug-and-play ADMM
% Input arguments:
% z0 = Initial point (primal variable)
% u0 = Initial point (dual variable)
% rho = Penalty parameter
% prox_f = Handle to function computing the inversion step
%        (proximal map of f/rho w.r.t. the appropriate scaling matrix)
% W = Handle to denoiser
% I = Ground-truth image to compare PSNR (optional), can be empty ([])
% tol = Tolerance (optional)
% maxiters = Max. no. of iterations (optional)
% objfun = Handle to function that computes objective value
%
% Output arguments:
% x_curr = Output (final iterate)
% converged = Flag, True if algorithm converged (error < tolerance)
% iters = No. of iterations completed
% psnr_vals = Iteration-wise PSNRs
% primal_residue = Iteration-wise values of norm(x_{k+1} - z_{k+1})
% dual_residue = Iteration-wise values of norm(z_{k+1} - z_k)
% obj = Iteration-wise objective function values
%

if(~exist('I','var') || isempty(I))
    calcPSNR = false;
else
    calcPSNR = true;
end
if(~exist('tol','var') || isempty(tol))
    tol = -1;   % Run until max. no. of iterations
end
if(~exist('maxiters','var') || isempty(maxiters))
    maxiters = 30;
end
if(~exist('objfun','var') || isempty(objfun))
    calcObj = false;
    obj = [];
else
    calcObj = true;
end

iters = 1;
z_curr = z0;
u_curr = u0;
dual_residue = nan(1,maxiters);
primal_residue = nan(1,maxiters);
if(calcPSNR)
    psnr_vals = nan(1,maxiters);
    psnr_vals(1) = psnr(z0,I,1);
end
if(calcObj)
    obj = nan(1,maxiters);
end
while(true)
    % Main algorithm
    x_next = prox_f(z_curr - u_curr,rho);
    v_next = x_next + u_curr;
    z_next = W(v_next);
    u_next = u_curr + x_next - z_next;
    
    % Calculate PSNR, residues, objective values etc.
    if(calcObj)
        obj(iters) = objfun(x_next,z_next,v_next);  % Calculate objective value
    end
    if(calcPSNR)
        psnr_vals(iters+1) = psnr(z_next,I,1);      % Calculate PSNR
        fprintf('Iteration = %d,   PSNR = %f\n',iters,psnr_vals(iters+1));
    else
        fprintf('Iteration = %d\n',iters);
    end
    err = euclNorm(z_next - z_curr);
    dual_residue(iters) = err;
    primal_residue(iters) = euclNorm(x_next - z_next);
    if(err < tol)
        converged = true;
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
    x_curr = x_next;
    z_curr = z_next;
    u_curr = u_next;
end

if(calcPSNR)
    psnr_vals(iters+1:end) = [];
else
    psnr_vals = [];
end

end


function n = euclNorm(y)
% Euclidean norm of vector or matrix

n = sqrt(sum(y(:).*y(:)));

end
