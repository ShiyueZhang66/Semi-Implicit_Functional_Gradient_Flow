import torch
import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from time import time
import random
import os

seed = 1234
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


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


###########################################################################################################
def semigwg(A,pdim, x0, score, kernel='gaussian', bw=1.,
               n_epoch=10000,f_iter=10, dim=64, latent_dim=120,step_size=1e-3,f_lr=1e-3, beta=.5,
               store=False, verbose=False):

    import torch.nn as nn
    import torch.optim as optim
    from tqdm import trange
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')


    class Small_F_net(nn.Module):

        def __init__(self, z_dim, latent_dim):
            super().__init__()
            self.z_dim = z_dim
            self.latent_dim = latent_dim
            self.dnn = nn.Sequential(nn.Linear(self.z_dim, self.latent_dim),
                                     nn.Tanh(),
                                     nn.Linear(self.latent_dim, self.latent_dim),
                                     nn.Tanh(),
                                     nn.Linear(self.latent_dim, self.z_dim))

        def forward(self, x):
            f = self.dnn(x)
            return f

    f_net = Small_F_net(dim, latent_dim).to(device)
    f_opt = optim.SGD(f_net.parameters(), lr=f_lr)


    def gssm(perturbed_samples,
             samples,score,
             init_semisigma=torch.tensor(0.1),
             n_particles=1,
             g_fn='p_norm',
             p=2,
             precond_alpha=1.0):
        dup_samples = samples.view(-1, dim)
        dup_samples.requires_grad_(True)
        dup_perturbed_samples = perturbed_samples.view(-1, dim)
        dup_perturbed_samples.requires_grad_(True)
        semi_sigm=init_semisigma
        semi_sigm.requires_grad_(True)

        f = f_net(dup_perturbed_samples)
        X_diff=dup_perturbed_samples-dup_samples

        loss= torch.mean(torch.norm((f+X_diff/ (semi_sigm ** 2)),dim=1,keepdim=True)**2)
        return loss

    # semi_sigma = torch.tensor(0.1)
    semi_sigma = torch.tensor(0.03)

    xs = []
    info = []

    x = x0.clone().to(device)

    for ep in trange(n_epoch):
        dup_x = x.data
        dup_x.requires_grad_(True)

        perturbed_x = dup_x + torch.randn_like(dup_x).to(device) * semi_sigma
        perturbed_x.requires_grad_(True)

        for i in range(f_iter):
            f_loss = gssm(perturbed_samples=perturbed_x, samples=dup_x,score=score, init_semisigma=semi_sigma)
            f_opt.zero_grad()
            f_loss.backward(retain_graph=True)
            f_opt.step()

        v = -(f_net(perturbed_x) - score(perturbed_x).to(device))
        x = x + step_size * v

    n_samples=x.shape[0]
    x_test=(x.reshape(n_samples,pdim, pdim)).detach().cpu().numpy()
    info = [amari_distance(y, A) for y in x_test]

    return info


#############################################################

def adasemigwg(A,pdim, x0, score, kernel='gaussian', bw=1.,
               n_epoch=10000,f_iter=10, dim=64, latent_dim=120,step_size=1e-3,f_lr=1e-3, beta=.5,
               store=False, verbose=False):

    import torch.nn as nn
    import torch.optim as optim
    from tqdm import trange
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')


    class Small_F_net(nn.Module):

        def __init__(self, z_dim, latent_dim):
            super().__init__()
            self.z_dim = z_dim
            self.latent_dim = latent_dim
            self.dnn = nn.Sequential(nn.Linear(self.z_dim, self.latent_dim),
                                     nn.Tanh(),
                                     nn.Linear(self.latent_dim, self.latent_dim),
                                     nn.Tanh(),
                                     nn.Linear(self.latent_dim, self.z_dim))

        def forward(self, x):
            f = self.dnn(x)
            return f

    f_net = Small_F_net(dim, latent_dim).to(device)
    f_opt = optim.SGD(f_net.parameters(), lr=f_lr)

    def gssm(perturbed_samples,
             samples,score,
             init_semisigma=torch.tensor(0.1),
             n_particles=1,
             g_fn='p_norm',
             p=2,
             precond_alpha=1.0):
        dup_samples = samples.view(-1, dim)
        dup_samples.requires_grad_(True)
        dup_perturbed_samples = perturbed_samples.view(-1, dim)
        dup_perturbed_samples.requires_grad_(True)
        semi_sigm=init_semisigma
        semi_sigm.requires_grad_(True)

        f = f_net(dup_perturbed_samples)
        X_diff=dup_perturbed_samples-dup_samples

        loss= torch.mean(torch.norm((f+X_diff/ (semi_sigm ** 2)),dim=1,keepdim=True)**2)
        return loss

    # semi_sigma=torch.tensor(0.1)
    semi_sigma = torch.tensor(0.03)

    semisigma_lr=torch.tensor(0.000001)

    xs = []
    info = []

    x = x0.clone().to(device)

    for ep in trange(n_epoch):

        dup_x = x.data
        dup_x.requires_grad_(True)

        noise=torch.randn_like(dup_x).to(device)

        perturbed_x = dup_x + noise* semi_sigma
        perturbed_x.requires_grad_(True)

        for i in range(f_iter):
            f_loss = gssm(perturbed_samples=perturbed_x, samples=dup_x,score=score, init_semisigma=semi_sigma)
            f_opt.zero_grad()
            f_loss.backward(retain_graph=True)
            f_opt.step()

        S = score(perturbed_x).to(device) - f_net(perturbed_x)
        semi_sigma_grad = torch.mean((S * noise).data)
        semi_sigma = semi_sigma + semisigma_lr * semi_sigma_grad
        v = -(f_net(perturbed_x) - score(perturbed_x).to(device))
        x = x + step_size * v



#################################################

    n_samples=x.shape[0]

    x_test=(x.reshape(n_samples,pdim, pdim)).detach().cpu().numpy()
    info = [amari_distance(y, A) for y in x_test]

    return info

###################

def gwg(A,pdim,x0, score, kernel='gaussian', bw=1.,
               n_epoch=10000,f_iter=10, dim=64, latent_dim=120,step_size=1e-3,f_lr=1e-3, beta=.5,
               store=False, verbose=False):

    import torch.nn as nn
    import torch.optim as optim
    from tqdm import trange
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')



    class Small_F_net(nn.Module):

        def __init__(self, z_dim, latent_dim):
            super().__init__()
            self.z_dim = z_dim
            self.latent_dim = latent_dim
            self.dnn = nn.Sequential(nn.Linear(self.z_dim, self.latent_dim),
                                     nn.Tanh(),
                                     nn.Linear(self.latent_dim, self.latent_dim),
                                     nn.Tanh(),
                                     nn.Linear(self.latent_dim, self.z_dim))

        def forward(self, x):
            f = self.dnn(x)
            return f

    f_net = Small_F_net(dim, latent_dim).to(device)
    f_opt = optim.SGD(f_net.parameters(), lr=f_lr)


    def gwg_gssm(samples,score,
             n_particles=1,
             g_fn='p_norm',
             p=2,
             precond_alpha=1.0):

        import torch.autograd as autograd

        dup_samples = samples.view(-1, dim)
        dup_samples.requires_grad_(True)

        score = score(dup_samples).to(device)

        f = f_net(dup_samples)

        loss1 = torch.sum(f * score, dim=-1).mean()
        loss2 = torch.zeros(samples.shape[0]).to(device)
        for _ in range(n_particles):
            vectors = torch.randn_like(dup_samples).to(device)
            gradv = torch.sum(f * vectors)
            grad2 = autograd.grad(gradv,
                                  dup_samples,
                                  create_graph=True,
                                  retain_graph=True)[0]
            loss2 += torch.sum(vectors * grad2, dim=-1) / n_particles
        loss2 = loss2.mean()

        if g_fn == 'p_norm':
            loss3 = torch.norm(f, p=p, dim=-1)**p / p
            loss3 = loss3.mean()
        # elif g_fn == 'precondition':
        #     loss3 = precondition_g(dup_samples, precond_alpha)

        loss = loss1 + loss2 - loss3
        return loss

    xs = []
    info = []

    x = x0.clone().to(device)

    for ep in trange(n_epoch):

        dup_x = x.data
        dup_x.requires_grad_(True)
        for i in range(f_iter):
            f_loss = -gwg_gssm(
                samples=dup_x,score=score)
            f_opt.zero_grad()
            f_loss.backward()
            f_opt.step()


        v = f_net(dup_x)
        x = x + step_size * v


    n_samples=x.shape[0]
    x_test=(x.reshape(n_samples,pdim, pdim)).detach().cpu().numpy()
    info = [amari_distance(y, A) for y in x_test]

    return info

################################################


def adapgwg(A,pdim,adap_lr,x0, score, kernel='gaussian', bw=1.,
               n_epoch=10000,f_iter=10, dim=64, latent_dim=120,step_size=1e-3,f_lr=1e-3, beta=.5,
               store=False, verbose=False):

    import torch.nn as nn
    import torch.optim as optim
    from tqdm import trange
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')



    class Small_F_net(nn.Module):

        def __init__(self, z_dim, latent_dim):
            super().__init__()
            self.z_dim = z_dim
            self.latent_dim = latent_dim
            self.dnn = nn.Sequential(nn.Linear(self.z_dim, self.latent_dim),
                                     nn.Tanh(),
                                     nn.Linear(self.latent_dim, self.latent_dim),
                                     nn.Tanh(),
                                     nn.Linear(self.latent_dim, self.z_dim))

        def forward(self, x):
            f = self.dnn(x)
            return f

    f_net = Small_F_net(dim, latent_dim).to(device)
    f_opt = optim.SGD(f_net.parameters(), lr=f_lr)


    def gwg_gssm(samples,score,p=2,
             n_particles=1,
             g_fn='p_norm',
             precond_alpha=1.0):

        import torch.autograd as autograd

        dup_samples = samples.view(-1, dim)
        dup_samples.requires_grad_(True)

        score = score(dup_samples).to(device)

        f = f_net(dup_samples)

        loss1 = torch.sum(f * score, dim=-1).mean()
        loss2 = torch.zeros(samples.shape[0]).to(device)
        for _ in range(n_particles):
            vectors = torch.randn_like(dup_samples).to(device)
            gradv = torch.sum(f * vectors)
            grad2 = autograd.grad(gradv,
                                  dup_samples,
                                  create_graph=True,
                                  retain_graph=True)[0]
            loss2 += torch.sum(vectors * grad2, dim=-1) / n_particles
        loss2 = loss2.mean()

        if g_fn == 'p_norm':
            loss3 = torch.norm(f, p=p, dim=-1)**p / p
            loss3 = loss3.mean()
        loss = loss1 + loss2 - loss3
        return loss

    xs = []
    info = []

    x = x0.clone().to(device)

    p=2.0

    for ep in trange(n_epoch):

        dup_x = x.data
        dup_x.requires_grad_(True)


        for i in range(f_iter):
            f_loss = -gwg_gssm(
                samples=dup_x,p=p,score=score)
            f_opt.zero_grad()
            f_loss.backward()
            f_opt.step()


        v = f_net(dup_x)
        x = x + step_size * v

        # adaptive p

        p_step =adap_lr
        lb = 1.1
        ub = 6.0

        scoredifference= abs(v).detach().cpu().numpy() ** p

        log_scoredifference = np.log(1e-7+1 / (abs(v).detach().cpu().numpy()  ** (p - 1)+1e-7))

        grad_p = ((1 / p ** 2) * np.sum(scoredifference).mean() - (1 / ((p - 1) ** 2 * p)) * np.sum(
            scoredifference * log_scoredifference).mean())
        grad_p = np.clip(grad_p, -0.1, 0.1)

        p += p_step * grad_p
        p = np.clip(p, lb, ub)

    n_samples=x.shape[0]
    x_test=(x.reshape(n_samples,pdim, pdim)).detach().cpu().numpy()
    info = [amari_distance(y, A) for y in x_test]

    return info

