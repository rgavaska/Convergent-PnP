function [g,winsize] = initInpainting(f,mask,initWinsize)
%INITINPAINTING Construct initial image for inpainting problem by applying
% Nan-exclusive median filtering (repeatedly if required).
% f = Input image with some pixels missing
% mask = Logical mask indicating missing pixels

if(~exist('initWinsize','var'))
    initWinsize = 1;
end

g = f;
g(mask) = nan;
winsize = initWinsize;
while(true)
    g = nanmedfilt2(g,winsize);
    if(nnz(isnan(g)) == 0)
        break;
    end
    winsize = winsize + 2;
end

end

