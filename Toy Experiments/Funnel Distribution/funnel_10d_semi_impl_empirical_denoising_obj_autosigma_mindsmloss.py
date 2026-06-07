import os
import math
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Distribution

device = torch.device(0 if torch.cuda.is_available() else 'cpu')
import time

import matplotlib.pylab as plt
import torch.nn as nn
from torch import autograd
from tqdm import trange
import ite
from torch.distributions.normal import Normal
from torch.distributions.studentT import StudentT
from torch.distributions.gamma import Gamma
import torch.nn.functional as F
from scipy.stats import laplace, multivariate_t, cauchy, gamma, cramervonmises
import random
import seaborn as sns

seed = 1254
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class target_funnel(object):
    name = 'target distribution (2D & 10D Funnel)'

    def __init__(self, sigma, dim, device="cpu"):
        self.dim = dim
        self.sigma = sigma
        self.device = device

    def sampler(self, n_sample):
        dominant = torch.randn((n_sample, 1)) * np.sqrt(self.sigma)
        other = torch.exp(dominant / 2) * torch.randn((n_sample, self.dim - 1))
        return (torch.cat((dominant, other), dim=1)).to(self.device)

    def log_prob(self, x):
        dominant = (self.dim - 1) * x[:, 0] / 2 + x[:, 0] ** 2 / (2 * self.sigma)
        other = torch.norm(x[:, 1:self.dim], dim=1) ** 2 * torch.exp(- x[:, 0]) / 2
        return - (dominant + other).reshape(x.shape[0], 1)

    def forward(self, x):
        dominant = x[:, 0] / self.sigma + (self.dim - 1) / 2 - torch.norm(x[:, 1:self.dim], dim=1) ** 2 * torch.exp(
            - x[:, 0]) / 2
        other = x[:, 1:self.dim] * torch.exp(- x[:, 0]).reshape(x.shape[0], 1)
        return - torch.cat((dominant.reshape(x.shape[0], 1), other), dim=1)

funnel=target_funnel(sigma=9,dim = 10)
sample = funnel.sampler(n_sample=1000).cpu().detach().numpy()

##################################################################################

def divergence_bf(dx, y):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()
def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1
def sample_gaussian_like(y):
    return torch.randn_like(y)
def divergence_approx(f, y, e=None):
    if e is None:
        e = sample_rademacher_like(y)
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx

################################################################################
    
class PFGfast:
  def __init__(self, P, net, optimizer1,optimizer2,optimizer3):
    self.P = P
    self.net = net
    self.optim1 = optimizer1
    self.optim2 = optimizer2
    self.optim3 = optimizer3
    self.p_update=p_update
  def phi(self, X):
    phi = self.net(X)
    return phi
  def step(self, X,X_ori,semi_sigma):
    #empirical method
    X = X.requires_grad_(True)
    log_prob = self.P.log_prob(X)
    score_func = torch.autograd.grad(log_prob.sum(), X)[0].detach()
    X_ori=X_ori.requires_grad_(True)
    estimate_grad = -torch.sum(semi_sigma ** (-10)*(X[:, None, :] - X_ori[None, :, :]) *
                              torch.exp(-torch.sum((X[:, None, :] - X_ori[None, :, :]) ** 2, dim=2)[:, :, None] / (
                                          2 * semi_sigma ** 2)),
                              dim=1) / torch.sum(
        semi_sigma ** (-10)*torch.exp(-torch.sum((X[:, None, :] - X_ori[None, :, :]) ** 2, dim=2) / (2 * semi_sigma ** 2)),
        dim=1, keepdim=True)/ (semi_sigma ** 2)
    X_ori.grad = estimate_grad-score_func
    self.optim1.step()
    return X_ori
  def score_step(self, X,X_ori, p_norm):
    H1 = torch.std(X,0)
    H1 = 1.0 / H1
    H1=torch.pow(H1, 0.2)
    H=torch.diag(H1)

    X = X.detach().cpu().requires_grad_(True)
    log_prob = self.P.log_prob(X)
    score_func = torch.autograd.grad(log_prob.sum(), X)[0].detach()
    
    self.net.train()
    X = X.to(device)
    S = self.net(X)

    self.optim2.zero_grad()
    score_func = score_func.to(device)
    X_diff=X-X_ori
    ##################################
    # denoising
    loss= torch.mean(torch.norm((S+X_diff/ semi_sigma ** 2),dim=1,keepdim=True)**2)

    ##############################
    loss.backward(retain_graph=True)
    self.optim2.step()


    #########################################################
    scoredifference = torch.abs(S) ** p_norm
    log_scoredifference = torch.log(1 / (torch.abs(S) ** (p_norm - 1)))

    stepsize = 0.00000025
    p_update=stepsize * ((1 / p_norm ** 2) * torch.sum(scoredifference) - (1 / ((p_norm - 1) ** 2 * p_norm)) * torch.sum(scoredifference * log_scoredifference)) / S.shape[0]

    if p_norm - p_update > 1.1:
        p_norm -= p_update
        p_norm = p_norm.item()
        return 1, p_norm

    else:
        return 0, p_norm

  def kl_distance(self,X_ori):
        KL = ite.cost.BDKL_KnnK()
        x_sample = X_ori.detach().cpu().numpy()
        x_true = funnel.sampler(n_sample=1000).detach().cpu().numpy()
        kl = KL.estimation(x_sample, x_true)
        return kl


###########################################################################

from torch import nn

n = 1000
d=10
t1 = time.time()
check_frq=100

X_0 = torch.randn(n, d)
X_0 = X_0.to(device)
h = 600

net = nn.Sequential(
    nn.Linear(d, h),
    nn.Tanh(),
    nn.Linear(h, h),
    nn.Tanh(),
    nn.Linear(h, d))
torch.save(net.state_dict(), "net.pth")

fig = plt.figure()
ax = fig.add_subplot()
p_list=[2.0]

Epoch=10001

semi_sigma_init=0.02
semi_sigma_0=torch.tensor(semi_sigma_init).to(device)

particles_lr=0.01
semisigma_lr=0

sigma_list=[]


for p in p_list:
    X=X_0.clone()
    semi_sigma=semi_sigma_0.clone()

    noise = torch.randn(n, d).to(device)
    Z = X + noise * semi_sigma

    net.load_state_dict(torch.load("net.pth"))
    net = net.to(device)

    optim1 = optim.SGD([X], lr=particles_lr, momentum=0.)
    optim2 = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, nesterov=1)
    optim3 = optim.SGD([semi_sigma], lr=semisigma_lr, momentum=0.)
    p_update=0

    pfg = PFGfast(funnel, net, optim1, optim2,optim3)

    p_0=p
    kl_list = np.zeros(Epoch // check_frq + 1)
    kl = pfg.kl_distance(X)
    kl_list[0] = kl

    count=0
    countlist = np.zeros(3)

    for i in range(Epoch):
        noise=torch.randn(n,d).to(device)
        Z = X + noise * semi_sigma
        X=pfg.step(Z,X,semi_sigma)

        if (i+1 ) % check_frq == 0:

            result_sample_Z =Z
            kl = pfg.kl_distance(result_sample_Z)

            kl_list[(i + 1) //
                    check_frq] = kl

            semi_sigma_cpu = semi_sigma.detach().cpu().numpy()
            sigma_list.append(semi_sigma_cpu)

t2 = time.time()
print(f'Computation Time: {t2-t1}')
