function [idx, obj, converge] = DFGOC(X, XTX, XTX_inv, W, alpha, beta, lambd, miu, c, L_dat, L_fea, m)
% https://github.com/qq774292747zxc/DFGOC
% the running code of DFGOC
% X: the data matrix which is n * d 
% XT: the transpose of X
% XTX_inv: the inverse matrix of XTX
% c: cluster's number
% L_dat, L_fea: the initialized Laplacian martix of data and feature space respectively
% m: the selected features' number
% W: the initialized projection matrix

t = 2; % fuziness degree
gama = 1e4; % orthogonal parameter
rng('default');
[n, d] = size(X);
v = ones(d, 1); % the feature selection vector
F = rand(c, c); % the indicator matrix
G = cmeans_initialization(X, n, c); % the clustering center
H = (G + abs(G))/2; % auxiliary matrix for non-negativity
maxIter = 21;
obj = zeros(1, maxIter);
obj(1) = inf;

Imc = eye(c);
Inc = zeros(n, c);
for i= 1:c
    Inc(i, i) = 1;
end
XvW = X * diag(v) * W;
converge = 1;
for iter_step = 2:maxIter
    % update G
    XvWF = XvW * F;
    PTZ = XvWF + gama*H;
    [UG, ~, VG] = svd(PTZ);
    G = UG * Inc * VG';
    % update H
    H = (G + abs(G))/2;
    % update W
    XTLX = alpha* X'* L_dat * X;
    vXTGFT = diag(v) * X' * G * F';
    t1 = XTLX + lambd*eye(d) + diag(v)*XTX*diag(v); %eye(d) + miu*eye(d)
    A = XTX_inv * t1;
    B = beta * L_fea + miu*eye(c);  % ensure positive definitiveness
    C = XTX_inv * vXTGFT;
    % W = lypa(A, B, -C);
    W = sylvester(A, B, C);
    
    % update v
    v = update_v(W*W', XTX, W*F*G'*X, d, m);

    % update F
    XvW = X * diag(v) * W;
    PTZ = G' * XvW;
    [UF, ~, VF] = svd(PTZ);
    F = VF * Imc * UF';
    
    % update S
    XW = X * W;
    [L_dat, ~] = updateLt(XW, t);
    WX = XW';
    [L_fea, ~] = updateLt(WX, t);
    % when c < k
    % [L_fea, ~] = updateLt_c(WX, t, c);
    
    t2 = XvW - G*F'; 
    WXLXW = WX * L_dat * XW;
    XWLWX = XW * L_fea * WX;
    obj(iter_step) = trace(t2*t2') + lambd*norm(W,'fro')^2 + alpha*trace(WXLXW) + beta*trace(XWLWX);
    %obj(iter_step)
    if obj(iter_step) < 1e-100
        break
    end
    t3 = obj(iter_step)-obj(iter_step-1);
    if t3 > 1
        converge = 0;
    end
    if abs(t3)/obj(iter_step) < 1e-3
        break
    end
end

[~, idx] = sort(v,'descend');
end                          
