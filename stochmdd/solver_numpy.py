import numpy as np
import time
from .mdc import MDC


class SGD():
    r"""Stochastic gradient

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

    def step(self, grad, **kwargs):
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


class SAG():
    r"""Stochastic average gradient

    Parameters
    ----------
    model : :obj:`np.ndarray`
        Model to update
    lr : :obj:`float`
        Learning rate (Note: in Appendix A of the SAG paper suggests having
        lr=n/L where L=n*eigmax(G^T G) where G is the linear operator of the
        problem and n is the number of terms in the finite sum)
    nfuncs : :obj:`int`
        Number of functions in finite-sum (when collecting multiple functions
        into one batch, this should be the number of batches)
    nbatches : :obj:`int`
        Number of batches per function
    bs : :obj:`int`
        Batch size
    weight_decay : :obj:`float`, optional
        Weight decay (Tikhonov regularization)
    saga : :obj:`bool`, optional
        Apply sag (``False``) or saga (``True``)
    adjusted : :obj:`bool`, optional
        Apply adjustment of first gradients until all batches have been sampled
        at least once

    """
    def __init__(self, model, lr, nfuncs, nbatches, bs, weight_decay=0.,
                 saga=False, adjusted=True):
        self.model = model
        self.lr = lr
        self.nfuncs = float(nfuncs)
        self.nbatches = float(nbatches)
        self.bs = bs
        self.weight_decay = weight_decay
        self.saga = self.nfuncs if saga else 1.
        self.adjusted = adjusted
        self.gradhistory = np.zeros((nbatches, model.size), model.dtype)
        self.gradcomputed = np.zeros(nbatches)
        self.gradfull = np.zeros_like(model)

    def step(self, grad, **kwargs):
        # Clever way for sag, does not work with saga
        if self.saga > 1:
            raise NotImplementedError('SAGA not available, use step1 or step2')
        # Update gradfull
        self.gradfull = self.gradfull + self.saga * (grad - self.gradhistory[kwargs['ibatch']].reshape(grad.shape))
        self.gradcomputed[kwargs['ibatch']] = 1.
        self.gradhistory[kwargs['ibatch']] = grad.ravel().copy()
        if self.adjusted:
            # Re-weight the early iterations
            if np.sum(self.gradcomputed) < self.nbatches:
                gradtoadd = self.gradfull * self.nfuncs / (self.bs * np.sum(self.gradcomputed))
            else:
                gradtoadd = self.gradfull
        else:
            gradtoadd = self.gradfull
        # Add weight decay
        if self.weight_decay > 0:
            gradtoadd += self.nfuncs * self.weight_decay * self.model
        # Update model and save gradient
        self.model -= (self.lr / self.nfuncs) * gradtoadd

    def step1(self, grad, **kwargs):
        ## Another way of implementing the same thing... simpler but less efficient
        ## for sag, but allows saga with minor change
        # Sum gradients
        d = np.sum(self.gradhistory, axis=0) / self.nfuncs
        gradtoadd = d + self.saga * (grad.ravel() - self.gradhistory[kwargs['ibatch']]) / self.nfuncs
        self.gradcomputed[kwargs['ibatch']] = 1.
        self.gradhistory[kwargs['ibatch']] = grad.ravel().copy()
        if self.adjusted:
            # Re-weight the early iterations
            if np.sum(self.gradcomputed) < self.nbatches:
                gradtoadd = gradtoadd * self.nfuncs / (self.bs * np.sum(self.gradcomputed))
        # Add weight decay
        if self.weight_decay > 0:
            gradtoadd += self.nfuncs * self.weight_decay * self.model.ravel()
        # Update model and save gradient
        self.model -= self.lr * gradtoadd.reshape(grad.shape)

    def step2(self, grad, **kwargs):
        ## Yet another way of implementing the same thing... from https://github.com/elmahdichayti/SAGA
        ## two different versions for sag and saga, same but more efficient than step1
        ## NOTE: Missing adjusted start!!
        if self.saga == 1:
            #SAG
            self.gradfull -= self.gradhistory[kwargs['ibatch']].reshape(grad.shape) / self.nfuncs
            self.gradcomputed[kwargs['ibatch']] = 1.
            self.gradhistory[kwargs['ibatch']] = grad.ravel().copy()
            self.gradfull += self.gradhistory[kwargs['ibatch']].reshape(grad.shape) / self.nfuncs
            gradtoadd = self.gradfull.copy()
            # Add weight decay
            if self.weight_decay > 0:
                gradtoadd += self.weight_decay * self.model
            # Update model and save gradient
            self.model -= self.lr * gradtoadd
        else:
            # SAGA
            gradtoadd = grad - self.gradhistory[kwargs['ibatch']].reshape(grad.shape) + self.gradfull
            self.gradcomputed[kwargs['ibatch']] = 1.
            self.gradfull += (grad - self.gradhistory[kwargs['ibatch']].reshape(grad.shape)) / self.nfuncs
            self.gradhistory[kwargs['ibatch']] = grad.ravel().copy()
            # Add weight decay
            if self.weight_decay > 0:
                gradtoadd += self.weight_decay * self.model
            # Update model and save gradient
            self.model -= self.lr * gradtoadd


class ExponentialLR():
    def __init__(self, optimizer, gamma=1.):
        self.optimizer = optimizer
        self.gamma = gamma
    def step(self):
        self.optimizer.lr *= self.gamma
