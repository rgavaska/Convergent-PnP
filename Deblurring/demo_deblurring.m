
clearvars; close all; clc;

%% Config setup
% Forward model parameters
ground_truth = './house.tif';   % Ground-truth image file
ker = fspecial('motion',10,45); % Blurring kernel
sigma_n = 10/255;               % Noise standard deviation (on a scale of [0,255])

% PnP restoration parameters
searchRad = 5;      % NLM search window radius
patchRad = 3;       % NLM patch radius
h = 25/255;         % NLM smoothing parameter
rho = 1;            % Regularization parameter (= reciprocal of step-size).
                    % Set rho to be greater than or equal to the Lipschitz
                    % constant of A'A, where A is the blurring operator.
                    % Since we are using a normalized blurring kernel, this
                    % Lipschitz constant is 1.
warm_iters = 5;     % No. of iterations in warm start
maxiters = 40;      % Max. no. of PnP-FISTA iterations AFTER initial warm-start

%% Generate corrupted observation
% Read ground-truth image
x_orig = im2double(imread(ground_truth));
if(size(x_orig,3)==3)
    x_orig = rgb2gray(x_orig);
end

% Forward model
[rr,cc] = size(x_orig);
b = imfilter(x_orig,ker,'circ');
b = b + sigma_n*randn(rr,cc);
b(b>1) = 1; b(b<0) = 0;

% Degradation function and gradient
filt_h = @(x) imfilter(x,ker,'circ');
gradf = @(x) filt_h(filt_h(x)-b);

%% Restoration
% Warm start to estimate a guide image:
% We run a small no. of iterations of PnP-FISTA in which we let the guide
% image be the same as the input image, and the scaling matrix D is
% computed from the output of the previous iteration. This makes the
% denoiser non-linear (since the guide image is not fixed). The image
% obtained at the end of these iterations is taken to be the guide image
% for the rest of the iterations, from which the NLM weights are computed.
% It is observed that we get better results if we use a higher value of the
% NLM smoothing parameter (h) during the warm start. Therefore, we scale
% the value of h by a factor of 10 for the warm start iterations.
x0 = b;
psnrs_warmstart = nan(1,warm_iters);
residues_warm = nan(1,warm_iters);
W_warm = @(x) JNLM(x,x,patchRad,searchRad,10*h);
fprintf('Running %d iterations as warm start\n',warm_iters);
for kk = 1:warm_iters
    [~,D_warm] = JNLM(x0,x0,patchRad,searchRad,10*h);
    D_warm = rho* D_warm/min(D_warm(:));
    x0_old = x0;
    x0 = pnpFISTA(x0,gradf,W_warm,D_warm,[],-1,1,[]);
    psnrs_warmstart(kk) = psnr(x0,x_orig,1);    % PSNR
    residues_warm(kk) = sqrt(sum((x0_old(:) - x0(:)).^2)); % Residue
    fprintf('Iteration = %d,\tPSNR = %f\n',kk,psnr(x0,x_orig,1));
end

% Main iterations with fixed denoiser matrix (= linear denoiser)
fprintf('\nRunning main %d iterations\n',maxiters);
W_nosym = @(x) JNLM(x,x0,patchRad,searchRad,h);     % Non-symmetric linear denoiser
[~,D] = JNLM(x0,x0,patchRad,searchRad,h);           % Normalizing weights
D = rho * D/min(D(:));                              % Scale D so that f is
                                                    % rho-smooth w.r.t. D
objfun = @(x,v) eval_fidelity_deblurring(x,filt_h,b) + ...
                  rho * eval_regularizer(x,v,D);    % Objective function
[x_hat,~,~,residues,psnr_vals,obj_vals] = ...
        pnpFISTA(x0,gradf,W_nosym,D,x_orig,[],maxiters,objfun);
psnr_vals = [psnrs_warmstart, psnr_vals];
residues = [residues_warm, residues];

%% Display images and plots
% Ground-truth, observed, and recovered images
figure('Units','Normalized','Position',[0.2,0.6,0.6,0.4]);
ax1 = subplot(1,3,1);
imshow(x_orig); title('Ground-truth image');
pause(0.1);     % Pause to prevent MATLAB from mixing up window titles
ax2 = subplot(1,3,2);
imshow(b); title('Observed image');
pause(0.1);
ax3 = subplot(1,3,3);
imshow(x_hat); title('Restored image');
drawnow;
margin = 0.001;
ax1.Position = [margin,0,0.33-2*margin,1];
ax2.Position = [0.33+margin,0,0.33-2*margin,1];
ax3.Position = [2*0.33+margin,0,0.33-2*margin,1];

% Plots
figure('Units','Normalized','Position',[0.1,0.0,0.8,0.4]);
ax1 = subplot(1,3,1);
plot(psnr_vals,'LineWidth',2.5);
grid on; axis tight;
xlabel('Iteration','Interpreter','latex');
title('PSNR','Interpreter','latex');
pause(0.1);

ax2 = subplot(1,3,2);
plot(0:length(residues)-1,log(residues),'LineWidth',2.5,'Color','g');
grid on; axis tight;
xlabel('Iteration','Interpreter','latex');
title('Errors, $\log(\| \mathbf{x}_{k+1} - \mathbf{x}_{k} \|_2)$',...
    'Interpreter','latex');
pause(0.1);

ax3 = subplot(1,3,3);
plot(obj_vals,'Linewidth',2.5);
grid on; axis tight;
xlabel('Iteration','Interpreter','latex');
title('Objective value, $f(\mathbf{x}_k)+g(\mathbf{x}_k)$',...
    'Interpreter','latex');

margin = 0.03;
ax1.Position = [margin,0.1,0.33-2*margin,1-0.2];
ax2.Position = [0.33+margin,0.1,0.33-2*margin,1-0.2];
ax3.Position = [2*0.33+margin,0.1,0.33-2*margin,1-0.2];
drawnow;
