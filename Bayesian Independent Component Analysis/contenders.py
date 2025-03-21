import numpy as np
import torch
from time import time
from scipy.optimize import fmin_l_bfgs_b

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


def svgd(A,pdim,x0, score, step, max_iter=1000, bw=1, tol=1e-5, verbose=False,
         store=False, backend='auto'):
    x_type = type(x0)
    if backend == 'auto':
        if x_type is np.ndarray:
            backend = 'numpy'
        elif x_type is torch.Tensor:
            backend = 'torch'
    if x_type not in [torch.Tensor, np.ndarray]:
        raise TypeError('x0 must be either numpy.ndarray or torch.Tensor '
                        'got {}'.format(x_type))
    if backend not in ['torch', 'numpy', 'auto']:
        raise ValueError('backend must be either numpy or torch, '
                         'got {}'.format(backend))
    if backend == 'torch' and x_type is np.ndarray:
        raise TypeError('Wrong backend')
    if backend == 'numpy' and x_type is torch.Tensor:
        raise TypeError('Wrong backend')
    if backend == 'torch':
        x = x0.detach().clone()
    else:
        x = np.copy(x0)
    n_samples, n_features = x.shape
    if store:
        storage = []
        timer = []
        t0 = time()

    xs = []
    info = []
    for i in range(max_iter):
        if store:
            if backend == 'torch':
                storage.append(x.clone())
            else:
                storage.append(x.copy())
            timer.append(time() - t0)
        d = (x[:, None, :] - x[None, :, :])
        dists = (d ** 2).sum(axis=-1)

        if backend == 'torch':
            k = torch.exp(- dists / bw / 2)
        else:
            k = np.exp(- dists / bw / 2)
        k_der = d * k[:, :, None] / bw
        scores_x = score(x)

        if backend == 'torch':
            ks = k.mm(scores_x)
        else:
            ks = k.dot(scores_x)
        kd = k_der.sum(axis=0)
        direction = (ks - kd) / n_samples

        criterion = (direction ** 2).sum()
        if criterion < tol ** 2:
            break

        x += step * direction
        if verbose and i % 100 == 0:
            print(i, criterion)

    x_test = (x.reshape(n_samples,pdim, pdim)).detach().cpu().numpy()
    info = [amari_distance(y, A) for y in x_test]

    return info

###################################################################


def gaussian_kernel(x, y, sigma):
    d = (x[:, None, :] - y[None, :, :])
    dists = (d ** 2).sum(axis=-1)
    return torch.exp(- dists / sigma / 2)


def mmd_lbfgs(x0, target_samples, bw=1, max_iter=10000, tol=1e-5,
              store=False):
    x = x0.clone().detach().numpy()
    n_samples, p = x.shape
    k_yy = gaussian_kernel(target_samples, target_samples, bw).mean().item()
    if store:
        class callback_store():
            def __init__(self):
                self.t0 = time()
                self.mem = []
                self.timer = []

            def __call__(self, x):
                self.mem.append(np.copy(x))
                self.timer.append(time() - self.t0)

            def get_output(self):
                storage = [torch.tensor(x.reshape(n_samples, p),
                                        dtype=torch.float32)
                           for x in self.mem]
                return storage, self.timer
        callback = callback_store()
    else:
        callback = None

    def loss_and_grad(x_numpy):
        x_numpy = x_numpy.reshape(n_samples, p)
        x = torch.tensor(x_numpy, dtype=torch.float32)
        x.requires_grad = True
        k_xx = gaussian_kernel(x, x, bw).mean()
        k_xy = gaussian_kernel(x, target_samples, bw).mean()
        loss = k_xx - 2 * k_xy + k_yy
        loss.backward()
        grad = x.grad
        return loss.item(), np.float64(grad.numpy().ravel())

    t0 = time()
    x, f, d = fmin_l_bfgs_b(loss_and_grad, x.ravel(), maxiter=max_iter,
                            factr=tol, epsilon=1e-12, pgtol=1e-10,
                            callback=callback)
    print('Took %.2f sec, %d iterations, loss = %.2e' %
          (time() - t0, d['nit'], f))
    output = torch.tensor(x.reshape(n_samples, p), dtype=torch.float32)
    if store:
        storage, timer = callback.get_output()
        return output, storage, timer
    return output
