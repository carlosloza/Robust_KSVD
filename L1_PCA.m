function U = L1_PCA(X)

max_it = 100;
conv_fl = 0;
tol = 10^-3;

[~, idx] = max(sum(X.^2, 1));
U_aux = X(:, idx);
U_ini = U_aux/norm(U_aux);
U_t = U_ini;

p = zeros(1, size(X,2));
it = 1;
while conv_fl == 0
    aux_t = U_t'*X;
    p(aux_t < 0) = -2;
    p = p + 1;
    U_t1 = sum(bsxfun(@times, p, X), 2);
    U_t1 = U_t1/norm(U_t1);
    aux_t1 = U_t1'*X;
    if norm(abs(U_t) - abs(U_t1)) <= tol
        conv_fl = 1;
        U = U_t1;
    elseif numel(find(aux_t1 == 0)) > 0
        display('This case!')
        rnd_U = 0.01*randn(size(X,1), 1);
        U_t1 = (U_t1 + rnd_U)/norm(U_t1 + rnd_U);
    elseif it == max_it
        display('Max Iterations')
        conv_fl = 1;
        U = U_t1;
    end
    U_t = U_t1;
    it = it + 1;    
end
end
