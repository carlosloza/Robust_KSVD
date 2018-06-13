% Function that implements L1-norm-based Principal Component Analysis
% Only first eigenvector (corresponding to largest eigenvalue) is
% estimated
% Code is based on the article "Principal Component Analysis Based on 
% L1-Norm Maximization" by N. Kwak. https://doi.org/10.1109/TPAMI.2008.114
% Author: Carlos Loza
% carlos85loza@gmail.com

function U = L1_PCA(X, max_it, tol)
% INPUTS:
% X : Matrix (n dimensions, N samples) to be decomposed
% max_it : Number of maximum iterations if convergence is not attained
% (positive integer). Default: 100
% tol : Convergence tolerance for successive estimated eigenvectors
% (positive floating point number). Default: 10e-3
% OUTPUT:
% U : Column vector (n dimensions). Resulting eigenvector.

switch nargin
    case 1
        max_it = 100;
        tol = 10e-3;
    case 2
        tol = 10e-3;
end

conv_fl = 1;                    % Convergence flag
% Initialization: Choose sample with largest L2-norm
[~, idx] = max(sum(X.^2, 1));
U_aux = X(:, idx);
U_ini = U_aux/norm(U_aux);
U_t = U_ini;

% Optimization
p = zeros(1, size(X,2));
it = 1;
while conv_fl
    aux_t = U_t'*X;
    p(aux_t < 0) = -2;
    p = p + 1;
    U_t1 = sum(bsxfun(@times, p, X), 2);
    U_t1 = U_t1/norm(U_t1);
    aux_t1 = U_t1'*X;
    if norm(abs(U_t) - abs(U_t1)) <= tol
        % Convergence achieved
        conv_fl = 0;
        U = U_t1;
    elseif numel(find(aux_t1 == 0)) > 0
        % Case introduced to avoid degenrate solutions
        rnd_U = 0.01*randn(size(X,1), 1);
        U_t1 = (U_t1 + rnd_U)/norm(U_t1 + rnd_U);
    elseif it == max_it
        % Maximum number of iterations
        display('Max Iterations')
        conv_fl = 0;
        U = U_t1;
    end
    U_t = U_t1;
    it = it + 1;    
end
end
