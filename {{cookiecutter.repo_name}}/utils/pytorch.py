import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_lrs(lr_sch_class, steps, **kwargs):
    test = torch.ones(1, requires_grad=True)
    opt = torch.optim.SGD([{'params': test, 'lr': 0.1}])
    sch = lr_sch_class(opt, **kwargs)

    has_batch_step = (hasattr(lr_sch_class, 'batch_step') and
                      callable(lr_sch_class.batch_step))
    lrs = []
    for _ in range(steps):
        sch.batch_step() if has_batch_step else sch.step()
        lrs.append(sch.get_lr())

    return np.array(lrs)


def plot_lrs(lr_sch_class, steps, by_epoch, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    lrs = compute_lrs(lr_sch_class, steps, by_epoch, **kwargs)
    ax.plot(range(steps), lrs)


def set_requires_grad(module, name, reqs_grad):
    for n, p in module.named_parameters():
        if n.startswith(name):
            p.requires_grad_(reqs_grad)


def set_requires_grads(module, **kwargs):
    for param, reqs_grad in kwargs.items():
        name = param.replace('__', '.')
        set_requires_grad(module, name, reqs_grad)
