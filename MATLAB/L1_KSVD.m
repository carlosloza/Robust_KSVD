% Function that implements robust dictionary learning via L1-norm-based
% dictionary update optimization (KSVD)
% Author: Carlos Loza
% carlos85loza@gmail.com

function [D, elap_time] = L1_KSVD(Y, K, L, n_it, sp_cod_m, D_ini)
% INPUTS:
% Y : Matrix (n dimensions, N samples) with an ideal sparse decomposition
% K : Number of dictionary atoms to be learned (positive integer)
% L : Sparsity support (positive integer)
% n_it : Number of alternating optimizations between Sparse Coding and 
% Dictionary Update (positive integer). Default = 25
% sp_cod_m : Sparse Coding algorithm (String). Default = 'OMP'
% D_ini : Matrix (n dimensions, K atoms). Initial dictionary. If not
% provided, it is initialized from data.
% OUTPUTS:
% D : Matrix (n dimensions, K atoms). Estimated dictionary.
% elap_time : Average processing time of Dictionary Update stage (positive
% floating point number)

N = size(Y, 2);

switch nargin
    case 3
        n_it = 25;
        sp_cod_m = 'OMP';
        % Initialize Dictionary
        D = Y(:,randperm(N,K));
        D = bsxfun(@rdivide,D,sqrt(sum(D.^2,1)));
    case 4
        sp_cod_m = 'OMP';
        % Initialize Dictionary
        D = Y(:,randperm(N,K));
        D = bsxfun(@rdivide,D,sqrt(sum(D.^2,1)));
    case 5
        % Initialize Dictionary
        D = Y(:,randperm(N,K));
        D = bsxfun(@rdivide,D,sqrt(sum(D.^2,1)));
    case 6
        D = D_ini;
end

elap_time_it = zeros(1, n_it);
for it = 1:n_it
    % Sparse Coding
    X = zeros(K,N);
    for i = 1:N
        [~, ~, alph_aux, idx_aux] = wmpalg(sp_cod_m,Y(:,i),D,'itermax',L);
        X(idx_aux,i) = alph_aux;
    end
    % Dictionary Update
    tic
    E = Y - D*X;
    for k = 1:K
        shrk_idx = find(X(k,:));
        if ~isempty(shrk_idx)
            Ek = E + D(:,k)*X(k,:);
            EkR = Ek(:,shrk_idx);
            if numel(shrk_idx) == 1
                D(:,k) = EkR/norm(EkR);
            else
                U = L1_PCA(EkR);
                D(:,k) = U;
            end
        end
    end
    elap_time_it(1, it) = toc;
end

elap_time = mean(elap_time_it);

end
