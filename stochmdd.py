import numpy as np
import torch
import torch.nn as nn
import pylops_gpu

from math import ceil
from torch.utils.data import TensorDataset, DataLoader
from pylops.waveeqprocessing.mdd import MDC as MDClops


class MDC(nn.Module):
    r"""Multi-dimensional convolution

    Wrap PyLops :py:func:`pylops.waveeqprocessing.mdd.MDC` as PyTorch module

    Parameters
    ----------
    nt : :obj:`int`
        Number of samples in time
    dt : :obj:`float`
        Sampling of time integration axis
    dr : :obj:`float`
        Sampling of receiver integration axis
    nv : :obj:`int`, optional
        Number of virtual receivers

    """
    def __init__(self, nt, dt, dr, nv=1, twosided=True, reciprocity=False):
        self.nt = nt
        self.dt = dt
        self.dr = dr
        self.nv = nv
        self.twosided = twosided
        if self.twosided:
            self.nteff = 2 * self.nt - 1
        else:
            self.nteff = self.nt
        self.reciprocity = reciprocity
        super().__init__()

    def forward(self, model, G):
        r"""Forward pass

        Parameters
        ----------
        m : :obj:`np.ndarray` (:math:`2n_t-1 \times n_r`)
            Model
        G : :obj:`np.ndarray`
            Frequency domain kernel (:math:`n_{f} \times n_s \times n_r`)

        """
        # Create operator
        ns = G.shape[1]
        MDCop = MDClops(G, nt=self.nteff, nv=self.nv, dt=self.dt,
                        dr=self.dr, twosided=self.twosided, transpose=False)
        MDCop = pylops_gpu.TorchOperator(MDCop, pylops=True)
        
        # Apply reciprocity
        if self.reciprocity:
            model = 0.5*(model + model.transpose(2, 1))
        
        # Apply operator to model data
        data = MDCop.apply(model.view(-1))
        data = data.view(self.nteff, ns, self.nv).squeeze()
        return data


def MDDbatch(nt, nr, dt, dr, Gfft, d, optimizer, n_epochs,
             twosided=True, mtrue=None, ivstrue=None,
             epochprint=10, **kwargs_solver):
    r"""MDD with batched gradient descent methods

    Parameters
    ----------
    nt : :obj:`int`
        Number of samples in time
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
    twosided : :obj:`bool`, optional
        Kernel is two-sided (``True``) or one-sided (``False``)
    mtrue : :obj:`torch.tensor`, optional
        True model (:math:`2n_t-1 \times n_r`)

    """

    # Create model to optimize for
    nteff = 2 * nt - 1 if twosided else nt
    nv = d.shape[-1] if len(d.shape) == 3 else 1
    model = torch.zeros((nteff, nr, nv),
                        dtype=d.dtype).squeeze().requires_grad_(True)

    # Define operator, loss, and solver
    criterion = nn.MSELoss()
    optimizer = optimizer([model], **kwargs_solver)
    MDCop = MDC(nt, dt, dr, nv=nv, twosided=twosided)
    
    losshist = []
    lossavg = []
    enormhist = []
    for epoch in range(n_epochs):
        # compute forward and loss
        data = MDCop(model, Gfft)
        loss = criterion(d, data)
        
        # backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # compute average loss over epochs
        losshist.append(loss.item())
        avg = sum(losshist) / len(losshist)
        lossavg.append(avg)

        # compute error norm
        if mtrue is not None:
            if ivstrue is None:
                enorm = np.linalg.norm(
                    model.detach().numpy() / model.detach().numpy().max() -
                    mtrue.detach().numpy() / mtrue.detach().numpy().max())
            else:
                enorm = np.linalg.norm(
                    model[..., ivstrue].detach().numpy() / model[..., ivstrue].detach().numpy().max() -
                    mtrue.detach().numpy() / mtrue.detach().numpy().max())
            enormhist.append(enorm)

        if (epoch + 1) % epochprint == 0:
            print(f'Epoch: {epoch + 1:3d}, loss : {loss.item():.4e}, loss avg : {avg:.4e}')

    # compute final data
    data = MDCop(model, Gfft)

    return model, data, losshist, lossavg, enormhist


def MDDminibatch(nt, nr, dt, dr, Gfft, d, optimizer, n_epochs, batch_size,
                 twosided=True, mtrue=None, ivstrue=None,
                 seed=None, scheduler=None, epochprint=10, reciprocity=False,
                 kwargs_sched=None, **kwargs_solver):
    r"""MDD with mini-batch gradient descent methods

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
        kwargs_sched : :obj:`dict`, optional
            Additional keyword arguments for scheduler
        kwargs_solver : :obj:`dict`, optional
            Additional keyword arguments for optimizer
            
    """
    # Set seed
    if seed is not None:
        torch.manual_seed(seed)

    # Create model to optimize for
    nteff = 2*nt-1 if twosided else nt
    nv = d.shape[-1] if len(d.shape) == 3 else 1
    model = torch.zeros((nteff, nr, nv),
                        dtype=d.dtype).squeeze().requires_grad_(True)

    # Reorganize data and kernel
    ns = Gfft.shape[1]
    Gfft = torch.transpose(Gfft, 1, 0)
    d = torch.transpose(d, 1, 0)

    # Define operator, loss, and solver
    criterion = nn.MSELoss()
    optimizer = optimizer([model], **kwargs_solver)
    MDCop = MDC(nt, dt, dr, nv=nv, twosided=twosided, reciprocity=reciprocity)
    MDCAllop = MDC(nt, dt, dr, nv=nv, twosided=twosided, reciprocity=reciprocity)

    # Define scheduler
    if scheduler is not None:
        scheduler = scheduler(optimizer, **kwargs_sched)

    # Create dataloader
    trainloader = DataLoader(dataset=TensorDataset(Gfft, d),
                             batch_size=batch_size,
                             shuffle=True)
    losshist = []
    lossavg = []
    lossepoch = []
    enormhist = []
    lr = []
    firstgrad = True
    #with torch.no_grad():
    #        dataall = MDCop(model, torch.transpose(Gfft, 1, 0).cpu().numpy())
    #        lossall = criterion(torch.transpose(dataall, 1, 0), d)
    #        lossepoch.append(lossall.item())
    lossepoch.append(criterion(d, d*0).item())

    for epoch in range(n_epochs):
        losses = []

        for Gbatch, dbatch in trainloader:
            optimizer.zero_grad()
            
            # compute forward and loss
            data = MDCop(model, torch.transpose(Gbatch, 1, 0).cpu().numpy())
            loss = criterion(torch.transpose(dbatch, 1, 0), data)

            # backward and optimize
            loss.backward()
            optimizer.step()
            
            # compute first gradient norm
            if firstgrad:
                with torch.no_grad():
                    firstgrad = False
                    grad = model.grad.clone().view(-1)
                    gnorm = torch.sum(torch.abs(grad)**2).item()
                    print('Initial Gradient norm: %e, scaled by lr: %e' % (gnorm, gnorm * optimizer.param_groups[0]["lr"]**2))
                    print('Initial Gradient norm as np.linalg.norm: %e, scaled by nbatches:  %e' % ((gnorm**.5*(nteff*batch_size*nv)/2), (gnorm**.5*(nteff*ns*nv)/2)))

            # update losses history
            losshist.append(loss.item())
            losses.append(loss.item())

            # compute error norm
            if mtrue is not None:
                if ivstrue is None:
                    enorm = np.linalg.norm(
                        model.detach().numpy() / model.detach().numpy().max() -
                        mtrue.detach().numpy() / mtrue.detach().numpy().max())
                else:
                    enorm = np.linalg.norm(
                        model[..., ivstrue].detach().numpy() / model[
                            ..., ivstrue].detach().numpy().max() -
                        mtrue.detach().numpy() / mtrue.detach().numpy().max())
                enormhist.append(enorm)

            # update learning rate
            if scheduler is not None:
                scheduler.step()
                lr.append(optimizer.param_groups[0]["lr"])

        # compute average loss over epoch
        avg_loss = sum(losses) / len(losses)
        lossavg.append(avg_loss)
        
        # compute loss of entire batch
        with torch.no_grad():
            dataall = MDCop(model, torch.transpose(Gfft, 1, 0).cpu().numpy())
            lossall = criterion(torch.transpose(dataall, 1, 0), d)
            lossepoch.append(lossall.item())

        if (epoch + 1) % epochprint == 0:
            print(f'epoch: {epoch + 1:3d}, loss : {loss.item():.4e}, loss avg : {avg_loss:.4e}')
    
    # compute final data
    data = MDCop(model, torch.transpose(Gfft, 1, 0).cpu().numpy())
    print('Final Model norm: %e' % torch.sum(torch.abs(model)**2).item())

    return model, data, losshist, lossavg, lossepoch, enormhist, lr


def MDDpage(nt, nr, dt, dr, Gfft, d, n_epochs, batch_size,
            batch_size1, lr, mtrue=None, seed=None, removelast=False,
            epochprint=10):

    # Set seed
    if seed is not None:
        torch.manual_seed(seed)

    # Create model to optimize for
    model = torch.zeros((2 * nt - 1, nr),
                        dtype=d.dtype).requires_grad_(True)

    # Reorganize data and kernel
    Gfft = torch.transpose(Gfft, 1, 0)
    d = torch.transpose(d, 1, 0)

    # Define operator, loss, and solver
    criterion = nn.MSELoss()
    MDCop = MDC(nt, dt, dr)

    # Create dataloader
    no_of_batches = ceil(Gfft.shape[1] / batch_size)
    trainloader = DataLoader(dataset=TensorDataset(Gfft, d),
                             batch_size=batch_size,
                             shuffle=True)
    if not removelast:
        trainloader1 = DataLoader(dataset=TensorDataset(Gfft, d),
                                 batch_size=batch_size1,
                                 shuffle=True)
    else:
        trainloader1 = DataLoader(dataset=TensorDataset(Gfft[:-1], d[:-1]),
                                  batch_size=batch_size1,
                                  shuffle=True)


    # Probability of gradient
    pt = batch_size1 / (batch_size + batch_size1)

    # First step
    probgtflag = True
    Gbatch, dbatch = next(iter(trainloader))
    data = MDCop(model, torch.transpose(Gbatch, 1, 0).cpu().numpy())
    loss = criterion(torch.transpose(dbatch, 1, 0), data)
    loss.backward()
    gt = model.grad.data

    losshist = []
    lossavg = []
    enormhist = []
    enormavg = []
    gradcount, grad1count = 0, 0
    for epoch in range(n_epochs):
        losses = []
        enorms = []

        for Gbatch, dbatch in trainloader:

            # save old model
            modelold = model.detach().clone().requires_grad_(True)

            # update model
            model.data.add_(-lr, gt)

            # save old gradient
            gtold = gt.data.detach().clone()

            # zero out gradients
            model.grad.data.zero_()

            # compute current loss
            with torch.no_grad():
                data = MDCop(model, torch.transpose(Gbatch, 1, 0).cpu().numpy())
                lossall = criterion(torch.transpose(dbatch, 1, 0), data)

            # random sample to choose with gradient step to use
            probgt = float(torch.rand(1)[0])
            probgtflag = probgt <= pt

            # compute new gradient
            if probgtflag:
                # gradient with large batch
                data = MDCop(model, torch.transpose(Gbatch, 1, 0).cpu().numpy())
                loss = criterion(torch.transpose(dbatch, 1, 0), data)
                loss.backward()
                gt = model.grad.data
                gradcount += 1

            else:
                # use adapted past gradient
                # 1. gradient with small batch and current model
                if grad1count % len(trainloader1) == 0:
                    tl1 = iter(trainloader1)
                Gbatch1, dbatch1 = next(tl1)
                data1 = MDCop(model, torch.transpose(Gbatch1, 1, 0).cpu().numpy())

                loss1 = criterion(torch.transpose(dbatch1, 1, 0), data1)
                # 2. gradient with small batch and old model
                data2 = MDCop(modelold, torch.transpose(Gbatch1, 1, 0).cpu().numpy())
                loss2 = criterion(torch.transpose(dbatch1, 1, 0), data2)
                loss1.backward()
                loss2.backward()
                gt = gtold + model.grad.data - modelold.grad.data
                grad1count += 1

            losshist.append(lossall.item())
            losses.append(lossall.item())

            # compute error norm
            if mtrue is not None:
                enorm = np.linalg.norm(
                    model.detach().numpy() / model.detach().numpy().max() -
                    mtrue.detach().numpy() / mtrue.detach().numpy().max())
                enormhist.append(enorm)
                enorms.append(enorm)

        # compute average loss and error norm over epoch
        avg_loss = sum(losses) / len(losses)
        lossavg.append(avg_loss)
        avg_enorm = sum(enorms) / len(enorms)
        enormavg.append(avg_enorm)

        if (epoch + 1) % epochprint == 0:
            print(f'epoch: {epoch + 1:3d}, loss : {lossall.item():.4e}, loss avg : {avg_loss:.4e}')

    # compute final data
    data = MDCop(model, torch.transpose(Gfft, 1, 0).cpu().numpy())

    print(f'grad: {gradcount}, grad1 : {grad1count}')

    return model, data, losshist, lossavg, enormhist, enormavg, gradcount, grad1count
