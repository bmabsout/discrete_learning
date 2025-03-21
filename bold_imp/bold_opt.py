import torch
from torch import Tensor


class BooleanOptimizer(torch.optim.Optimizer):

    def __init__(self, params, lr: float):
        super(BooleanOptimizer, self).__init__(params, dict(lr=lr))
        for param_group in self.param_groups:
            param_group['accums'] = [torch.zeros_like(p.data) for p in param_group['params']]
            param_group['ratios'] = [0 for p in param_group['params']] 
        self._nb_flips = 0

    @property
    def nb_flips(self):
        n = self._nb_flips
        self._nb_flips = 0
        return n

    def step(self):
        for param_group in self.param_groups:
            for idx, p in enumerate(param_group['params']):
                self.update(p, param_group, idx)

    def update(self, param: Tensor, param_group: dict, idx: int):
        # Assert that gradient values are integers
        assert torch.all(torch.eq(param.grad.data, torch.round(param.grad.data))), "Gradients must contain only integer values"
        accum = param_group['ratios'][idx] * param_group['accums'][idx] + param_group['lr'] * param.grad.data
        param_group['accums'][idx] = accum
        #print(param.grad.data.mean(),accum.mean())
        param_to_flip = accum * (2 * param.data - 1) >= 1
        param.data[param_to_flip] = torch.logical_not(param.data[param_to_flip]).float()
        param_group['accums'][idx][param_to_flip] = 0.
        param_group['ratios'][idx] = 1 - param_to_flip.float().mean()
        self._nb_flips += float(param_to_flip.float().sum())

class BoldVanillaOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr: float, thresh: int):
        super(BoldVanillaOptimizer, self).__init__(params, dict(lr=lr))
        self._nb_flips = 0
        self.thresh = thresh
    @property
    def nb_flips(self):
        n = self._nb_flips
        self._nb_flips = 0
        return n

    def step(self):
        for param_group in self.param_groups:
            for idx, p in enumerate(param_group['params']):
                self.update(p, param_group, idx)

    def update(self, param: Tensor, param_group: dict, idx: int):
        # Assert that gradient values are integers
        assert torch.all(torch.eq(param.grad.data, torch.round(param.grad.data))), "Gradients must contain only integer values"
        # Assert that weights are binary (0 or 1)
        assert torch.all(torch.logical_or(param.data == 0, param.data == 1)), "Weights must contain only binary values (0 or 1)"

        # update rule based on that in paper w <- not w if XNOR(G, W) == T

        # ver.0 vanilla rule: w <- not w if (w == T and grad > 0) or (w == F and grad < 0)
        # param_to_flip = (param.data == 1) * (param.grad.data > 0 ) + (param.data == 0) * (param.grad.data < 0)
        # this does not work. It flips at the beginning and quickly cease to update

        # ver.1 vanilla with threshold: w <- not w if (w == T and grad > thresh) or (w == F and grad < -thresh)
        param_to_flip = (param.data == 1) * (param.grad.data > self.thresh) + (param.data == 0) * (param.grad.data < -self.thresh)
        
        
        param.data[param_to_flip] = torch.logical_not(param.data[param_to_flip]).float()
        self._nb_flips += float(param_to_flip.float().sum())