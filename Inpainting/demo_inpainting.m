
clearvars; close all; clc;

%% Config setup
% Forward model parameters
ground_truth = './peppers.tif';  % Ground-truth image file
r = 0.5;            % Fraction of missing pixels
sigma_n = 20/255;   % Noise standard deviation (on a scale of [0,255])

% PnP restoration parameters
searchRad = 5;      % NLM search window radius
patchRad = 3;       % NLM patch radius
h = 45/255;         % NLM smoothing parameter
rho = 1;            % Regularization parameter
maxiters = 100;     % Max. no. of PnP-ADMM iterations

%% Generate corrupted observation
% Read ground-truth image
x_orig = im2double(imread(ground_truth));
if(size(x_orig,3)==3)
    x_orig = rgb2gray(x_orig);
end

% Forward model
[rr,cc] = size(x_orig);
A = double(rand(rr,cc)>=r);     % Binary inpainting mask
b = A.*x_orig;
b(A==1) = b(A==1) + sigma_n*randn(nnz(A),1);
b(b>1) = 1; b(b<0) = 0;

%% Restoration
% Initialization
z0 = initInpainting(b,A==0,5);                     % ADMM initial point
W_nosym = @(x) JNLM(x,z0,patchRad,searchRad,h);    % Non-symmetric linear denoiser
[~,~,~,~,D] = JNLM(z0,z0,patchRad,searchRad,h);    % Normalizing weights
prox_map = @(x,r) prox_inpainting(r,D,A,b,x);      % Proximal map
objfun = @(x,z,v) eval_fidelity_inpainting(x,A,b) +...
         rho * eval_regularizer(z,v,D);            % Objective function

% Main algorithm
tic;
[x_hat,~,~,psnr_vals,primal_res,dual_res,obj_vals] = ...
    pnpADMM(z0,0,rho,prox_map,W_nosym,x_orig,[],maxiters,objfun);
toc;

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
plot(0:length(primal_res)-1,log(primal_res),'LineWidth',2.5,'Color','g',...
    'DisplayName','$\log(\mathbf{x}_{k+1} - \mathbf{z}_{k+1})$');
hold on;
plot(0:length(dual_res)-1,log(dual_res),'LineWidth',2.5,'Color','m',...
    'DisplayName','$\log(\mathbf{z}_{k+1} - \mathbf{z}_k)$');
grid on; hold off; axis tight;
xlabel('Iteration','Interpreter','latex');
title('Errors','Interpreter','latex');
legend('show','Interpreter','latex');
pause(0.1);

ax3 = subplot(1,3,3);
plot(obj_vals,'Linewidth',2.5);
grid on; axis tight;
xlabel('Iteration','Interpreter','latex');
title('Objective value, $f(\mathbf{x}_k)+g(\mathbf{z}_k)$',...
    'Interpreter','latex');

margin = 0.03;
ax1.Position = [margin,0.1,0.33-2*margin,1-0.2];
ax2.Position = [0.33+margin,0.1,0.33-2*margin,1-0.2];
ax3.Position = [2*0.33+margin,0.1,0.33-2*margin,1-0.2];
drawnow;
