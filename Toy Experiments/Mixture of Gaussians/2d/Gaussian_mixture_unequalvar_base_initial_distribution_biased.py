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


seed = 1244
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



import torch.distributions as D
K = 5
torch.manual_seed(1)
mix = D.Categorical(torch.ones(K,).to(device))
comp = D.Independent(D.Normal(
    (torch.randn(K,2)*2).to(device), torch.tensor([[0.1,0.1],[0.2,0.2],[0.3,0.3],[0.4,0.4],[0.5,0.5]]).to(device)), 1)
gmm = D.MixtureSameFamily(mix, comp)
sample = gmm.sample((2000,)).cpu()

plt.plot(sample[:,0],sample[:,1],'.')
plt.show()


############
# groundtruth=gmm.sample((1000,)).cpu()
#
# fig, ax = plt.subplots(figsize=(5, 5))
# bbox = [-5, 5, -5, 5]
# xx, yy = np.mgrid[bbox[0]:bbox[1]:100j, bbox[2]:bbox[3]:100j]
# positions = np.vstack([xx.ravel(), yy.ravel()])
# f = -np.log(-np.reshape(gmm.log_prob(torch.Tensor(positions.T).to(device)).cpu().numpy(), xx.shape))
#
# cxx, cyy = np.mgrid[bbox[0]:bbox[1]:30j, bbox[2]:bbox[3]:30j]
# ax.axis(bbox)
# ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))
# cfset = ax.contourf(xx, yy, f, cmap='Blues', alpha=0.8, levels=11)
# groundtruth_plot=groundtruth
# ax.plot(groundtruth_plot[:, 0], groundtruth_plot[:, 1], '.', markersize=2, color='#ff7f0e')
#
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# # ax.set_title(f"Multimodal-{i+1}-L2GF", fontsize=20, y=1.04)
# ax.set_title(f"Groundtruth", fontsize=20, y=1.04)
#
# fig.tight_layout()
# plt.subplots_adjust(wspace=0.1, hspace=0.05)
# plt.figure_format = 'retina'
# plt.savefig(f'../mog_unequalvar_2d_results/2d_groundtruth.pdf', bbox_inches='tight', pad_inches=0.1)

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
  def __init__(self, P, net, optimizer1,optimizer2):
    self.P = P
    self.net = net
    self.optim1 = optimizer1
    self.optim2 = optimizer2
    self.p_update=p_update
  def phi(self, X):
    phi = self.net(X) / X.size(0)
    return -phi
  def step(self, X):
    self.optim1.zero_grad()
    X.grad = self.phi(X)
    self.optim1.step()
  def score_step(self, X, p_norm):
    H1 = torch.std(X,0)
    H1 = 1.0 / H1
    H1=torch.pow(H1, 0.2)
    H=torch.diag(H1)

    X = X.requires_grad_(True)
    log_prob = self.P.log_prob(X)
    score_func = torch.autograd.grad(log_prob.sum(), X)[0].detach()

    self.net.train()
    X = X.to(device)
    S = self.net(X)
    self.optim2.zero_grad()
    score_func = score_func.to(device)

    # H_2
    # loss = (-torch.sum(score_func*S) - torch.sum(divergence_approx(S,X)) + 0.5*torch.trace(H.matmul(S.T).matmul(S)))/S.shape[0]

    # lp
    score_func=score_func.to(device)
    loss = (-torch.sum(score_func * S) - torch.sum(divergence_approx(S, X)) + torch.norm(S, p=p_norm)**p_norm /p_norm   )/ S.shape[0]


    loss.backward()
    self.optim2.step()
    scoredifference = torch.abs(S) ** p_norm
    log_scoredifference = torch.log(1 / (torch.abs(S) ** (p_norm - 1)))

    stepsize = 0.00000005
    p_update=stepsize * ((1 / p_norm ** 2) * torch.sum(scoredifference) - (1 / ((p_norm - 1) ** 2 * p_norm)) * torch.sum(scoredifference * log_scoredifference)) / S.shape[0]

    # p_adam
    # self.p_update=stepsize *((1 / p_norm ** 2) * torch.sum(scoredifference) - (1 / ((p_norm - 1) ** 2 * p_norm)) * torch.sum(scoredifference * log_scoredifference)) / S.shape[0]

    if p_norm - p_update > 1.1:
        p_norm -= p_update
        p_norm = p_norm.item()
        return 1, p_norm

    else:
        return 0, p_norm

  def kl_distance(self):
        KL = ite.cost.BDKL_KnnK()
        x_sample = X.detach().cpu().numpy()
        x_true = gmm.sample((1000,)).detach().cpu().numpy()
        kl = KL.estimation(x_sample, x_true)
        return kl


###########################################################################

from torch import nn

n = 1000
t1 = time.time()

check_frq=10

initvar=0.5

X_0 = torch.randn(n, 2)
X_0 = torch.randn(n, 2)*initvar+torch.tensor([3,0])
X_0 = X_0.to(device)
# X_0=np.loadtxt(f'../mog_unequalvar_2d_results/2d_l2gf_0_samples_fortraj_sameseed_1244_biased_var_{initvar}.txt')
# X_0=torch.tensor(X_0,dtype=torch.float32)
# X_0 = X_0.to(device)

h = 32
net = nn.Sequential(
    nn.Linear(2, h),
    nn.Tanh(),
    nn.Linear(h, h),
    nn.Tanh(),
    nn.Linear(h, 2))
torch.save(net.state_dict(), "net.pth")

fig = plt.figure()
ax = fig.add_subplot()
p_list=[2.0]

Epoch=2300

for p in p_list:
    X=X_0.clone()
    X_plot = X.detach().cpu().numpy()
    # np.savetxt(
    #     # f"../mog_unequalvar_2d_results/2d_adagwg_0_samples_fortraj_sameseed_1244_biased_var_{initvar}.txt",
    #     f"../mog_unequalvar_2d_results/2d_l2gf_0_samples_fortraj_sameseed_1244_biased_var_{initvar}.txt",
    #     X_plot)
    net.load_state_dict(torch.load("net.pth"))
    net = net.to(device)

    # optim1 = optim.SGD([X], lr=5e-1, momentum=0.)#l2gf
    optim1 = optim.SGD([X], lr=5e-1, momentum=0.)  # adagwg
    optim2 = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, nesterov=1)
    p_update=0

    pfg = PFGfast(gmm, net, optim1, optim2)

    p_0=p
    n = 1000
    kl_list = np.zeros(Epoch // check_frq + 1)
    kl = pfg.kl_distance()
    kl_list[0] = kl

    for i in range(10):
        pfg.score_step(X, p)

    count=0
    countlist = np.zeros(3)

    for i in range(Epoch):
        for j in range(5):
            if i > 5: #adaptive p
            # if i > 50000000: #non-adaptive p
                flag, p_nm = pfg.score_step(X, p)
                p = p_nm
            else:
                flag, p_nm = pfg.score_step(X, p)

        pfg.step(X)

        if (i + 1) % check_frq == 0:
            kl = pfg.kl_distance()
            kl_list[(i + 1) //
                    check_frq] = kl

            X_plot = X.detach().cpu().numpy()

            # np.savetxt(
            #     # f"../mog_unequalvar_2d_results/2d_adagwg_{i + 1}_samples_fortraj_sameseed_1244_biased_var_{initvar}.txt",
            #     f"../mog_unequalvar_2d_results/2d_l2gf_{i + 1}_samples_fortraj_sameseed_1244_biased_var_{initvar}.txt",
            #     X_plot)

            # if (i+1)==1600:
            #     np.savetxt(
            #         f"../mog_unequalvar_2d_results/2d_l2gf_{i+1}_samples.txt",
            #         X_plot)
            #
            # if (i+1)==2000:
            #     np.savetxt(
            #         f"../mog_unequalvar_2d_results/2d_l2gf_{i+1}_samples.txt",
            #         X_plot)

    print(kl_list)
    print(count)
    print(p)
    print(countlist)

ax.legend()
# plt.savefig('particle_vi_mog_kl.jpg', dpi=600)
plt.show()

t2 = time.time()
print(f'Computation Time: {t2-t1}')
