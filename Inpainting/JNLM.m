function [DenoisedImg, A1, M_nsy, Y_swp,Z] = JNLM(NoisyImg, GuideImg,...
                            PatchSizeHalf, WindowSizeHalf, h)
% Fast "joint" nonlocal means filter for image denoising.
% Identical to standard NLM except that the weights are computed from
% a "guide" image that may be distinct from the noisy image.
% This code is a slight modification of another matlab code that implements
% a fast version of standard NLM. 
% Unfortunately I do not know any online source for the standard NLM code.


%NLM_II  
% Get Image Info
[Height,Width] = size(NoisyImg);
[M1,N] = size(NoisyImg);
S = WindowSizeHalf;
K = PatchSizeHalf;
A = zeros ( N^2, (2*S+1)^2 );       %zero initialization
M_nsy = zeros ( N^2, (2*S+1)^2 );   %zero initialization 
Y_swp = zeros ( N^2, (2*S+1)^2 );   %zero initialization
A1 = zeros ( N^2, (2*S+1)^2 );       %zero initialization

% Initialize the denoised image
u = zeros(Height,Width); 
% Initialize the weight max
M = u; 
% Initialize the accumlated weights
Z = M;


% Pad noisy image to avoid Borader Issues
PaddedGuideImg = padarray(GuideImg,[PatchSizeHalf,PatchSizeHalf],'symmetric','both');
%PaddedImg1 = padarray(NoisyImg,[K+S, K+S]);
PaddedV = padarray(NoisyImg,[WindowSizeHalf,WindowSizeHalf],'symmetric','both');
% B=padarray(NoisyImg,[(K+S),(K+S)],'symmetric'); %zero-oadding

% Main loop
for dx =  -S:S  %-WindowSizeHalf :WindowSizeHalf
    for dy = -S:S   %-WindowSizeHalf :WindowSizeHalf
        %if dx ~= 0 || dy ~= 0
        % Compute the Integral Image 
        [Sd, diff, t] = integralImgSqDiff(PaddedGuideImg,dx,dy);
        Hat = triangle(dx, dy, S);
        size(Hat);
        % Obtaine the Square difference for every pair of pixels
      %  SqDist = Sd(PatchSizeHalf+1:end-PatchSizeHalf,PatchSizeHalf+1:end-PatchSizeHalf) + Sd(1:end-2*PatchSizeHalf,1:end-2*PatchSizeHalf) - Sd(1:end-2*PatchSizeHalf,PatchSizeHalf+1:end-PatchSizeHalf)-Sd(PatchSizeHalf+1:end-PatchSizeHalf,1:end-2*PatchSizeHalf);       
%         PaddedImg
%         B = circshift(PaddedImg, [-dx,- dy])
%         
%   t;
%         t-B
%         
%         
%         SqDiff1 = (PaddedImg - B).^2
%         diff
%         SqDiff1 - diff

        temp1 = img2DShift(Sd, K, K);
        temp2 = img2DShift(Sd, -K-1,- K-1);     % integralImgSqDiff(Sd, K, K);
        temp3 = img2DShift(Sd, -(K+1), K) ;    %integralImgSqDiff(Sd, 0, K);
        temp4 = img2DShift(Sd, K, -(K+1))  ;  %integralImgSqDiff(Sd, K, 0);
        Res = temp1 + temp2 -temp3 -temp4;
%         size(Res)
%         Res(1:3, K+1:N+K)
%         SqDist1 = Res(K+S+1:M1+K+S, K+S+1:N+K+S)
        SqDist1 = Res(K+1 : M1+K, K+1 : N+K);

        % Compute the weights for every pixels
      %     w = exp(-SqDist1/(h^2));
          w = Hat.*exp(-SqDist1/(h^2));

%         w = exp(-SqDist/(2*Sigma^2));


                row = (1+dx+S); %j=dx
                col = (1+dy+S); %k=dy
                row1 = (1+dx+S);
                col1 = (1+dy+S);
               % A( :, (col-1)*(2*S+1) + row )= reshape(w, Height*Width, 1);
                A1( :, (col1-1)*(2*S+1) + row1 )= (reshape(w', Height*Width, 1))';
                
%                 M_nsy( (i-1)*N + l, (col-1)*(2*S+1) + row ) = B((S+K+i+j),(S+K+l+k));
%                 Y_swp( (i-1)*N + l, (col-1)*(2*S+1) + row ) = B((S+K+i-j),(S+K+l-k));               

        % Obtaine the corresponding noisy pixels
        v = PaddedV((WindowSizeHalf+1+dx):(WindowSizeHalf+dx+Height),(WindowSizeHalf+1+dy):(WindowSizeHalf+dy+Width));
        v_swp = PaddedV((WindowSizeHalf+1-dx):(WindowSizeHalf-dx+Height),(WindowSizeHalf+1-dy):(WindowSizeHalf-dy+Width));
        
        M_nsy( :, (col-1)*(2*S+1) + row ) = reshape(v', Height*Width, 1)';
        Y_swp( :, (col-1)*(2*S+1) + row ) = reshape(v_swp', Height*Width, 1)';
                
       % Compute and accumalate denoised pixels
       % v = NoisyImg;
        u = u+w.*v;
        % Update weight max
        % M = max(M,w);
        % Update accumlated weighgs
        Z = Z+w;
       % end
    end
end

% % Speical controls to accumlate the contribution of the noisy pixels to be denoised        
% f = 1;
% u = u+f*M.*NoisyImg;
% u = u./(Z+f*M);
u = u./Z;
% Output denoised image
% Wt_matrix = A;
DenoisedImg = u;



function [Sd, diff, t] = integralImgSqDiff(v,dx,dy)
% FUNCTION intergralImgDiff: Compute Integral Image of Squared Difference
    % Decide shift type, tx = vx+dx; ty = vy+dy
    t = img2DShift(v,dx,dy);

    % Create sqaured difference image
    diff = (v-t).^2;
    % Construct integral image along rows
    Sd = cumsum(diff,1);
    % Construct integral image along columns
    Sd = cumsum(Sd,2);

function hat = triangle(dx, dy, Ns)
%     nu = abs(1 - abs(dx)) * abs(1 - abs(dy));
%     dn = (Ns+1)^2;
    r1 = abs(1 - abs(dx)/(Ns + 1));
    r2 = abs(1 - abs(dy)/(Ns + 1)); % * abs(1 - abs(dy));
    
    hat = r1 * r2;



function t = img2DShift(v,dx,dy)
% FUNCTION img2DShift: Shift Image with respect to x and y coordinates
    t = zeros(size(v));
    type = (dx>0)*2+(dy>0);
    switch type
        case 0 % dx<0,dy<0: move lower-right 
            t(-dx+1:end,-dy+1:end) = v(1:end+dx,1:end+dy);
        case 1 % dx<0,dy>0: move lower-left
            t(-dx+1:end,1:end-dy) = v(1:end+dx,dy+1:end);
        case 2 % dx>0,dy<0: move upper-right
            t(1:end-dx,-dy+1:end) = v(dx+1:end,1:end+dy);
        case 3 % dx>0,dy>0: move upper-left
            t(1:end-dx,1:end-dy) = v(dx+1:end,dy+1:end);

    end



% -------------------------------------------------------------------------
