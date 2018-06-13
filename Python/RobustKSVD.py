""" Robust Dictionary Learning algorithms
"""
# Author: Carlos Loza
# carlos85loza@gmail.com

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.decomposition import TruncatedSVD
import time


def dct_basis(N):
    D = np.zeros((N, N))
    for k in range(0, N):
        if k == 0:
            coef = np.sqrt(1 / N)
        else:
            coef = np.sqrt(2 / N)
        n = np.arange(0, N)
        D[:, k] = coef * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    return D


def add_awgn(x, n_dB):
    aux = np.var(x)/(10 ** (n_dB/10))
    no = np.random.randn(x.shape[0])
    no_add = np.sqrt(aux) * no
    y = x + no_add
    return y


def gen_sparse_samp(D, L, N):
    n = D.shape[0]
    Y = np.zeros((n, N))
    for i in range(0, N):
        aux = np.zeros(n)
        D_index = np.random.randint(D.shape[1], size=L)
        for j in range(0, L):
            D_aux = D[:, D_index[j]]
            aux += np.multiply(np.random.rand(1), D_aux)
        Y[:, i] = aux
    return Y


def gen_sparse_samp_impulsive_noise(D, L, N, r_out, SNR_imp):
    out_samp = np.random.choice(N, np.around(r_out * N).astype(int), replace=False)
    n = D.shape[0]
    Y = np.zeros((n, N))
    for i in range(0, N):
        aux = np.zeros(n)
        D_index = np.random.randint(D.shape[1], size=L)
        for j in range(0, L):
            D_aux = D[:, D_index[j]]
            aux += np.multiply(np.random.rand(1), D_aux)
        if np.any(out_samp == i):
            Y[:, i] = add_awgn(aux, SNR_imp)
        else:
            Y[:, i] = aux
    return Y


def ksvd(Y, K, L, *args):
    N = Y.shape[1]

    if len(args) == 0:
        n_it = 25
        D = Y[:, np.random.randint(N, size=K)]
        D /= np.sqrt(np.sum((D ** 2), axis=0))
    elif len(args) == 1:
        n_it = args[0]
        D = Y[:, np.random.randint(N, size=K)]
        D /= np.sqrt(np.sum((D ** 2), axis=0))
    elif len(args) == 2:
        n_it = args[0]
        D = args[1]

    n = D.shape[0]
    elap_time_it = np.zeros(n_it)
    for it in range(0, n_it):
        # Sparse Coding
        #X = np.zeros((K, N))
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=L, tol=None, fit_intercept=False)
        omp.fit(D, Y)
        X = np.transpose(omp.coef_)
        # Dictionary Update
        start_t = time.time()
        E = Y - np.matmul(D, X)
        for k in range(0, K):
            shrk_idx = np.asarray(np.nonzero(X[k, :]))
            if shrk_idx.size > 0:
                Ek = E + np.outer(D[:, k], X[k, :])
                # Check this
                EkR = np.squeeze(Ek[:, shrk_idx])
                if shrk_idx.size == 1:
                    D[:, k] = EkR/np.linalg.norm(EkR, 2)
                else:
                    tsvd = TruncatedSVD(1, algorithm="arpack")
                    tsvd.fit(np.transpose(EkR))
                    D[:, k] = tsvd.components_
        end_t = time.time()
        elap_time_it[it] = end_t - start_t
    elap_time = np.mean(elap_time_it)
    return D, elap_time


def l1_pca(X, max_it = 100, tol = 10e-3):
    conv_fl = 1
    idx = np.argmax(np.sum((X ** 2), axis=0))
    U_ini = X[:, idx]/np.linalg.norm(X[:, idx], 2)
    U_t = U_ini
    p = np.zeros(X.shape[1])
    it = 1
    while conv_fl:
        aux_t = np.dot(U_t, X)
        p[aux_t < 0] = -2
        p = p + 1
        U_t1 = np.sum((p * X), axis=1)
        U_t1 = U_t1/np.linalg.norm(U_t1, 2)
        aux_t1 = np.dot(U_t1, X)
        if np.linalg.norm((np.absolute(U_t) - np.absolute(U_t1)), 2) <= tol:
            conv_fl = 0
            U = U_t1
        elif np.count_nonzero(aux_t1) < p.shape[0]:
            rnd_U = 0.01*np.random.randn(X.shape[0].astype(int))
            U_t1 = (U_t1 + rnd_U)/np.linalg.norm((U_t1 + rnd_U), 2)
        elif it == max_it:
            print("Max Iterations")
            conv_fl = 0
            U = U_t1
        U_t = U_t1
        it += 1

    return U


def l1_ksvd(Y, K, L, *args):
    N = Y.shape[1]

    if len(args) == 0:
        n_it = 25
        D = Y[:, np.random.randint(N, size=K)]
        D /= np.sqrt(np.sum((D ** 2), axis=0))
    elif len(args) == 1:
        n_it = args[0]
        D = Y[:, np.random.randint(N, size=K)]
        D /= np.sqrt(np.sum((D ** 2), axis=0))
    elif len(args) == 2:
        n_it = args[0]
        D = args[1]

    n = D.shape[0]
    elap_time_it = np.zeros(n_it)
    for it in range(0, n_it):
        # Sparse Coding
        #X = np.zeros((K, N))
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=L, tol=None, fit_intercept=False)
        omp.fit(D, Y)
        X = np.transpose(omp.coef_)
        # Dictionary Update
        start_t = time.time()
        E = Y - np.matmul(D, X)
        for k in range(0, K):
            shrk_idx = np.asarray(np.nonzero(X[k, :]))
            if shrk_idx.size > 0:
                Ek = E + np.outer(D[:, k], X[k, :])
                # Check this
                EkR = np.squeeze(Ek[:, shrk_idx])
                if shrk_idx.size == 1:
                    D[:, k] = EkR/np.linalg.norm(EkR, 2)
                else:
                    D[:, k] = l1_pca(EkR)
        end_t = time.time()
        elap_time_it[it] = end_t - start_t
    elap_time = np.mean(elap_time_it)
    return D, elap_time
