"""
=======================================
Bayesian Independent Component Analysis
=======================================
"""  # noqa

# Author: Pierre Ablin <pierre.ablin@ens.fr>
#
# License: MIT

import torch
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


############################################
meg_data = np.loadtxt('MEG_art.txt')
meg_data=torch.tensor(meg_data,dtype=torch.float32)
meg_data_10=meg_data[0:5,:]
meg_data_10=meg_data_10.T
############################################


#hmc

p_list =5
sigma = 10
bw = 0.1


def kinetic_energy(velocity):
    return 0.5 * torch.norm(velocity)**2

def hamiltonian(position, velocity, energy_function):
    return energy_function(position) + kinetic_energy(velocity)

def leapfrog_step(x0,
                  v0,
                  score,
                  step_size,
                  num_steps):

    # Start by updating the velocity a half-step
    v = v0 - 0.5 * step_size * score(x0)

    # Initalize x to be the first step
    x = x0 + step_size * v

    for i in range(num_steps):
        # Compute gradient of the log-posterior with respect to x
        gradient = score(x)

        # Update velocity
        v = v - step_size * gradient

        # Update x
        x = x + step_size * v

    # Do a final update of the velocity for a half step
    v = v - 0.5 * step_size * score(x)

    # return new proposal state
    return x, v

def hmc(initial_x,
        step_size,
        num_steps,
        score,energy):
    """Summary

    Parameters
    ----------
    initial_x : tf.Variable
        Initial sample x ~ p
    step_size : float
        Step-size in Hamiltonian simulation
    num_steps : int
        Number of steps to take in Hamiltonian simulation
    log_posterior : str
        Log posterior (unnormalized) for the target distribution

    Returns
    -------
    sample :
        Sample ~ target distribution
    """

    v0 = torch.randn_like(initial_x)
    x, v = leapfrog_step(initial_x,
                      v0,
                      step_size=step_size,
                      num_steps=num_steps,
                      score=score)

    orig = hamiltonian(initial_x, v0, energy)
    current = hamiltonian(x, v, energy)
    p_accept = min(1.0, np.exp(orig - current))

    if p_accept > np.random.uniform():
        ind = 1
        return x,p_accept,ind
    else:
        ind = 0
        return initial_x,p_accept,ind







def meg_hmc(p, sigma):
    W = sigma * np.random.randn(p, p)
    A = np.linalg.pinv(W)
    X = meg_data_10.T
    size_train=17730
    n = 17730

    hmc_iter = 100000
    # n_samples=10
    n_samples = 1

    hmc_samples =torch.zeros(hmc_iter, p**2)


    def logcosh(x):
        s = np.sign(x) * x
        p = np.exp(-2 * s)
        return torch.tensor(s + np.log1p(p) - np.log(2))



    def score(w):
        N, _ = w.shape
        w_list = w.reshape(N, p, p)
        w_list=w_list
        z = w_list.matmul(X)
        psi = torch.tanh(z).matmul(meg_data_10) / n - torch.inverse(
            w_list
        ).transpose(-1, -2)
        sc = psi + w_list / sigma**2
        return sc.reshape(N, p ** 2)

    def energy(w):
        N, _ = w.shape
        w_list = w.reshape(N, p, p)
        w_list=w_list
        z = w_list.matmul(X)

        psi = -torch.sum(-torch.log(torch.cosh(z))) / n - torch.log(
            torch.abs(torch.det(w_list))
        )
        sc = psi + torch.norm(w_list)**2 / (2*sigma**2)
        return sc

    x = torch.randn(n_samples, p ** 2)*sigma

    acce = 0

    for t in trange(hmc_iter):
        step_size=5e-4
        num_steps=40
        x,p_accept,ind = hmc(x, step_size,num_steps,score,energy)
        acce = acce + ind
        print(acce / (t + 1), 'avg_rate')
        print(p_accept,'accept')
        hmc_samples[t]=x

    return hmc_samples,step_size

hmc_result,step_size=meg_hmc(p=p_list,sigma=sigma)
torch.save(hmc_result,f'hmc_result_dim5_100000_correct_{step_size}_100000samples.pt')