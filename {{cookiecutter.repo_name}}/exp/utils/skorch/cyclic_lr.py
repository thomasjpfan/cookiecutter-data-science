"""Contains learning rate scheduler callbacks"""

# pylint: disable=unused-import
import numpy as np
from torch.optim.optimizer import Optimizer


class CyclicLR(object):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    ``batch_step`` should be called after a batch has been used for training.
    To resume training, save `last_batch_idx` and use it to instantiate
    ``CycleLR``.

    This class has three built-in policies, as put forth in the paper:

    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each
        cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations)
        at each cycle iteration.

    This implementation was adapted from the github repo:
    `bckenstler/CLR <https://github.com/bckenstler/CLR>`_

    Parameters
    ----------
    optimizer : torch.optimizer.Optimizer instance.
      Optimizer algorithm.

    base_lr : float or list of float (default=1e-3)
      Initial learning rate which is the lower boundary in the
      cycle for each param groups (float) or each group (list).

    max_lr : float or list of float (default=6e-3)
      Upper boundaries in the cycle for each parameter group (float)
      or each group (list). Functionally, it defines the cycle
      amplitude (max_lr - base_lr). The lr at any cycle is the sum
      of base_lr and some scaling of the amplitude; therefore max_lr
      may not actually be reached depending on scaling function.

    step_size : int (default=2000)
      Number of training iterations per for first half of a cycle.
      Authors suggest setting step_size 2-8 x training iterations in epoch.

    step_size_2 : int (default=None)
      Number of training iterations per for second half of a cycle.
      Authors suggest setting step_size 2-8 x training iterations in epoch.

    mode : str (default='triangular')
      One of {triangular, triangular2, exp_range}. Values correspond
      to policies detailed above. If scale_fn is not None, this
      argument is ignored.

    gamma : float (default=1.0)
      Constant in 'exp_range' scaling function:
      gamma**(cycle iterations)

    scale_fn : function (default=None)
      Custom scaling policy defined by a single argument lambda
      function, where 0 <= scale_fn(x) <= 1 for all x >= 0.
      mode paramater is ignored.

    scale_mode : str (default='cycle')
      One of {'cycle', 'iterations'}. Defines whether scale_fn
      is evaluated on cycle number or cycle iterations (training
      iterations since start of cycle).

    last_batch_idx : int (default=-1)
      The index of the last batch.

    Examples
    --------

    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> scheduler = torch.optim.CyclicLR(optimizer)
    >>> data_loader = torch.utils.data.DataLoader(...)
    >>> for epoch in range(10):
    >>>     for batch in data_loader:
    >>>         scheduler.batch_step()
    >>>         train_batch(...)

    References
    ----------

    .. [1] Leslie N. Smith, 2017, "Cyclical Learning Rates for
        Training Neural Networks,". "ICLR"
        `<https://arxiv.org/abs/1506.01186>`_

    """

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, step_size_2=None, mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle',
                 last_batch_idx=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.base_lrs = self._format_lr('base_lr', optimizer, base_lr)
        self.max_lrs = self._format_lr('max_lr', optimizer, max_lr)

        step_size_2 = step_size_2 or step_size
        self.total_size = float(step_size + step_size_2)
        self.step_ratio = float(step_size) / self.total_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_idx + 1)
        self.last_batch_idx = last_batch_idx

    def _format_lr(self, name, optimizer, lr):
        """Return correctly formatted lr for each param group."""
        if isinstance(lr, (list, tuple)):
            if len(lr) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(lr)))
            return np.array(lr)
        else:
            return lr * np.ones(len(optimizer.param_groups))

    def step(self, epoch=None):
        """Not used by ``CyclicLR``, use batch_step instead."""
        pass

    def batch_step(self, batch_idx=None):
        """Updates the learning rate for the batch index: ``batch_idx``.
        If ``batch_idx`` is None, ``CyclicLR`` will use an internal
        batch index to keep track of the index.
        """
        if batch_idx is None:
            batch_idx = self.last_batch_idx + 1
        self.last_batch_idx = batch_idx
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    # pylint: disable=unused-argument
    def _triangular_scale_fn(self, x):
        """Cycle amplitude remains contant"""
        return 1.

    def _triangular2_scale_fn(self, x):
        """
        Decreases the cycle amplitude by half after each period,
        while keeping the base lr constant.
        """
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        """
        Scales the cycle amplitude by a factor ``gamma**x``,
        while keeping the base lr constant.
        """
        return self.gamma**(x)

    def get_lr(self):
        """Calculates the learning rate at batch index:
        ``self.last_batch_idx``.
        """
        cycle = np.floor(1 + self.last_batch_idx / self.total_size)
        x = 1 + self.last_batch_idx / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = 1/(self.step_ratio-1)*(x-1)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_idx)
            lrs.append(lr)
        return lrs
