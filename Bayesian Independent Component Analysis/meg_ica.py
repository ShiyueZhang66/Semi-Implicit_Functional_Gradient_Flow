"""
=======================================
Bayesian Independent Component Analysis
=======================================
"""  
import torch
from contenders import svgd
from utils import semigwg,gwg,adasemigwg,adapgwg
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from tqdm import trange


seed = 5432
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True




meg_data = np.loadtxt('MEG_art.txt')
meg_data=torch.tensor(meg_data,dtype=torch.float32)
meg_data_10=meg_data[0:5,:]
meg_data_10=meg_data_10.T

def amari_distance(W, A):
    """
    Computes the Amari distance between two matrices W and A.
    It cancels when WA is a permutation and scale matrix.
    Parameters
    ----------
    W : ndarray, shape (n_features, n_features)
        Input matrix
    A : ndarray, shape (n_features, n_features)
        Input matrix
    Returns
    -------
    d : float
        The Amari distance
    """
    P = np.dot(W, A)

    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)

    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])

#################################################

def one_expe(n, p,i, sigma, bw, n_samples,adap_lr):
    W_ori = torch.load('hmc_result_dim5_100000_correct_0.0005_100000samples.pt')
    W = W_ori[50000:100000]
    W=torch.mean(W,dim=0)


    W=W.reshape(p, p)
    A = np.linalg.pinv(W)
    S = np.random.laplace(size=(p, n))

    X = meg_data_10.T

    def score(w):
        N, _ = w.shape
        w_list = w.reshape(N, p, p)
        w_list=w_list
        z = w_list.matmul(X)
        psi = torch.tanh(z).matmul(X.t()) / n - torch.inverse(
            w_list
        ).transpose(-1, -2)
        sc = -psi - w_list / sigma**2
        return sc.reshape(N, p ** 2)

    def score_gwg(w):
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        N, _ = w.shape
        w_list = w.reshape(N, p, p)
        w_list = w_list.detach().cpu()
        z = w_list.matmul(X)
        psi = torch.tanh(z).matmul(X.t()) / n - torch.inverse(
            w_list
        ).transpose(-1, -2)
        sc = -psi - w_list / sigma**2
        return sc.reshape(N, p ** 2)

    x = torch.randn(n_samples, p ** 2)*sigma

#####################################################################################
    # amari_svgd= svgd(A,p, x.clone(), score, 0.1, bw=bw, max_iter=2000)
    amari_svgd= svgd(A, p, x.clone(), score, 0.03, bw=bw, max_iter=2000)

    amari_semigwg= semigwg(A,p, x.clone(), score_gwg, n_epoch=2000, f_iter=20, dim=p ** 2,f_lr=1e-3)

    amari_adasemigwg= adasemigwg(A, p, x.clone(), score_gwg, n_epoch=2000, f_iter=20, dim=p ** 2, f_lr=1e-3)

    amari_gwg= gwg(A,p, x.clone(), score_gwg, n_epoch=2000, f_iter=20, dim=p ** 2,f_lr=1e-3)

    f_lr=1e-3
    amari_adapgwg= adapgwg(A, p,adap_lr, x.clone(), score_gwg, n_epoch=2000, f_iter=20, dim=p ** 2, f_lr=f_lr)  # better

#################################################################

    # np.savetxt(
    #     f'meg_ica_results_post_samples_comparison/hmc_result_dim5_100000_correct_0.0005_50000post_amari_svgd_ep2000_samp_{n_samples}_stepsize_0.1_{i}.txt',
    #     amari_svgd)
    #
    # np.savetxt(
    #     f'meg_ica_results_post_samples_comparison/hmc_result_dim5_100000_correct_0.0005_50000post_amari_semigwg_ep2000_samp_{n_samples}_semisigma_0.03_ada_1e-6_{i}.txt',
    #     amari_adasemigwg)
    #
    # np.savetxt(
    #     f'meg_ica_results_post_samples_comparison/hmc_result_dim5_100000_correct_0.0005_50000post_amari_semigwg_ep2000_samp_{n_samples}_semisigma_0.03_{i}.txt',
    #     amari_semigwg)
    #
    # np.savetxt(
    #     f'meg_ica_results_post_samples_comparison/hmc_result_dim5_100000_correct_0.0005_50000post_amari_gwg_ep2000_samp_{n_samples}_{i}.txt',
    #     amari_gwg)
    #
    # np.savetxt(
    #     f'meg_ica_results_post_samples_comparison/hmc_result_dim5_100000_correct_0.0005_50000post_amari_adapgwg_p_adap{adap_lr}_f_lr{f_lr}_ep2000_samp_{n_samples}_{i}.txt',
    #     amari_adapgwg)


    return (
        amari_semigwg,
        amari_adasemigwg,
        amari_gwg,
        amari_adapgwg,
        amari_svgd
    )


####################################################

p_list = [5]

n = meg_data_10.shape[0]

sigma = 10
check_frq=50
bw = 0.1

n_samples = 100
n_tries = 5

# n_samples = 10
# n_tries = 50

d_save = {}
for p in p_list:
    print(p)
    d_save[p] = {}
    amari_semigwgs = []
    amari_gwgs = []
    amari_ksds = []
    amari_svgds = []
    amari_randoms = []
    x_semigwgs = []
    x_gwgs = []
    x_ksds = []
    x_svgds=[]
    x_randoms = []

    # adap_lr=0.008
    adap_lr = 0.004

    for i in range(n_tries):
        (
            amari_semigwg,
            amari_adasemigwg,
            amari_gwg,
            amari_adapgwg,
            amari_svgd
        ) = one_expe(n, p, i,sigma, bw, n_samples,adap_lr=adap_lr)

