function [D, elap_time] = L1_KSVD(Y, K, L, n_it, sp_cod_m, D_ini)

[n,N] = size(Y);

if nargin == 5
    % Initialize Dictionary
    D = Y(:,randperm(N,K));
    D = bsxfun(@rdivide,D,sqrt(sum(D.^2,1)));
elseif nargin == 6
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
