import numpy as np
import time
from .mdc import MDC


class SGD():
    r"""Stochastic gradient descent

    Parameters
    ----------
    model : :obj:`np.ndarray`
        Model to update
    lr : :obj:`float`
        Learning rate
    weight_decay : :obj:`float`, optional
        Weight decay (Tikhonov regularization)
    momentum : :obj:`float`, optional
        Momentum
    nesterov : :obj:`bool`, optional
        Use Nesterov momentum

    """
    def __init__(self, model, lr, weight_decay=0.,
                 momentum=0., nesterov=False):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.firststep = True

    def step(self, grad):
        # Add weight decay
        if self.weight_decay > 0:
            grad += self.weight_decay * self.model
        # Add momentum
        if self.momentum > 0.:
            if self.firststep:
                self.b = grad
                self.firststep = False
            else:
                self.b = self.momentum * self.b + grad
            if self.nesterov:
                grad = grad + self.momentum * self.b
            else:
                grad = self.b
        # Update model and save gradient
        self.model -= self.lr * grad


class ExponentialLR():
    def __init__(self, optimizer, gamma=1.):
        self.optimizer = optimizer
        self.gamma = gamma
    def step(self):
        self.optimizer.lr *= self.gamma


def MDDminibatch(nt, nr, dt, dr, Gfft, d, optimizer, n_epochs, batch_size, shuffle=True,
                 twosided=True, mtrue=None, ivstrue=None, enormabsscaling=False,
                 seed=None, scheduler=None, epochprint=10, reciprocity=False,
                 savegradnorm=False, savefirstgrad=False, timeit=True,
                 kwargs_sched=None, **kwargs_solver):
    r"""MDD with mini-batch gradient descent methods

    Note that all norms used in print statements have been normalized to be
    in agreement with torch implementation

    Parameters
    ----------
    nt : :obj:`int`
        Number of samples in time
    nr : :obj:`int`
        Number of samples in receiver axis
    dt : :obj:`float`
        Sampling of time integration axis
    dr : :obj:`float`
        Sampling of receiver integration axis
    Gfft : :obj:`np.ndarray`
        Frequency domain kernel (:math:`n_{f} \times n_s \times n_r`)
    d : :obj:`torch.tensor`
        Data (:math:`2n_t-1 \times n_s`)
    optimizer : :obj:`torch.optimizer`
        Optimizer function handle`
    n_epochs : :obj:`int`
        Number of samples in time
    batch_size : :obj:`int`
        Size of batch
    shuffle : :obj:`bool`, optional
        Shuffle before batching
    twosided : :obj:`bool`, optional
        Kernel is two-sided (``True``) or one-sided (``False``)
    mtrue : :obj:`torch.tensor`, optional
        True model (:math:`2n_t-1 \times n_r`)
    ivstrue : :obj:`int`, optional
        Index of virtual source to select when  computing error norm
    seed : :obj:`int`, optional
        Seed (if set, the data will be shuffled always in the same way
    scheduler : :obj:`torch.optim.lr_scheduler`, optional
        Scheduler object
    epochprint : :obj:`int`, optional
        Number of epochs after which the losses are printed on screen
    reciprocity : :obj:`bool`, optional
        Enfore reciprocity at each iteration
    savegradnorm : :obj:`bool`, optional
        Save norm of gradient over iterations
    savegradnorm : :obj:`bool`, optional
        Time solver
    savefirstgrad : :obj:`bool`, optional
        Save first gradientkwargs_sched : :obj:`dict`, optional
        Additional keyword arguments for scheduler
    timeit : :obj:`bool`, optional
        Time execution
    kwargs_solver : :obj:`dict`, optional
        Additional keyword arguments for optimizer

    """
    if timeit:
        t0 = time.time()

    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Create model to optimize for
    nteff = 2*nt-1 if twosided else nt
    ns = Gfft.shape[1]
    nv = d.shape[-1] if len(d.shape) == 3 else 1
    model = np.zeros((nteff, nr, nv), dtype=d.dtype).squeeze()

    # Define operator
    MDCop = MDC(Gfft, nt=nteff, nv=nv, dt=dt, dr=dr, twosided=twosided,
                fast=False)

    # Define optimizer
    optimizer = optimizer(model, **kwargs_solver)

    # Define scheduler
    if scheduler is not None:
        scheduler = scheduler(optimizer, **kwargs_sched)

    # Optimize
    losshist = []
    lossavg = []
    lossepoch = []
    enormhist = []
    lr = []
    gnormhist = []
    firstgrad = True

    for epoch in range(n_epochs):
        losses = []
        isrcs = np.arange(ns)
        if shuffle:
            np.random.shuffle(isrcs)
        for ibatch in range(int(np.ceil(ns / batch_size))):
            # Select sources batch
            isrcbatch = isrcs[ibatch * batch_size:(ibatch + 1) * batch_size]
            MDCop.update(isrcbatch)

            # Compute gradient
            grad, loss = MDCop.grad(d[:, isrcbatch].ravel(), model)

            # Compensate for last gradient that may be smaller than batch_size
            if len(isrcbatch) < batch_size:
                grad *= (batch_size / len(isrcbatch))

            # Update model
            optimizer.step(grad.reshape(model.shape))

            # Compute gradient norm
            if firstgrad:
                gnorm = np.linalg.norm(grad / ((2 * nt - 1) * nr)) ** 2
                print('Initial Loss norm: %e' % (loss / ((2 * nt - 1) * batch_size)))
                print('Initial Gradient norm: %e, scaled by lr: %e' % (gnorm, gnorm * optimizer.lr ** 2))
                gradfirst = grad.copy()
                firstgrad = False

            # Update losses history
            losses.append(loss)
            losshist.append(loss)

            # Compute error norm
            if mtrue is not None:
                if ivstrue is None:
                    if enormabsscaling:
                        mmax = np.abs(model).max()
                        mtruemax = np.abs(mtrue).max()
                    else:
                        mmax = model.max()
                        mtruemax = mtrue.max()
                    enorm = np.linalg.norm(model / mmax -
                                           mtrue / mtruemax)
                else:
                    if enormabsscaling:
                        mmax = np.abs(model[..., ivstrue]).max()
                        mtruemax = np.abs(mtrue).max()
                    else:
                        mmax = model[..., ivstrue].max()
                        mtruemax = mtrue.max()
                    enorm = np.linalg.norm(model[..., ivstrue] / mmax -
                                           mtrue/ mtruemax)
                enormhist.append(enorm)

            # Update learning rate
            if scheduler is not None:
                scheduler.step()
                lr.append(optimizer.lr)

        # Compute average loss over epoch
        avg_loss = sum(losses) / len(losses)
        lossavg.append(avg_loss)

        if (epoch + 1) % epochprint == 0:
            print(f'epoch: {epoch + 1:3d}, loss : {loss.item() / ((2 * nt - 1) * batch_size):.4e}, loss avg : {avg_loss / ((2 * nt - 1) * batch_size):.4e}')

    # Compute final data
    MDCop.update(np.arange(ns))
    data = (MDCop @ model.ravel()).reshape(nteff, ns, nv).squeeze()

    if timeit:
        print('Time: %f s' % (time.time() - t0))

    if not savefirstgrad:
        return model, data, losshist, lossavg, lossepoch, enormhist, lr
    else:
        return model, data, losshist, lossavg, lossepoch, enormhist, gradfirst
