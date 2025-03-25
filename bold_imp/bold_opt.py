import torch
from torch import Tensor
from utils import get_tensor_stats


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
        get_tensor_stats(param.grad.data)
        
        param.data[param_to_flip] = torch.logical_not(param.data[param_to_flip]).float()
        self._nb_flips += float(param_to_flip.float().sum())

class BoldVanillaMomentumOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr: float, thresh: int, momentum: float = 0.9, dampening: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening)
        super(BoldVanillaMomentumOptimizer, self).__init__(params, defaults)
        self._nb_flips = 0
        self.thresh = thresh
        
        # Initialize momentum buffers for each parameter
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['momentum_buffer'] = torch.zeros_like(p.data)
    
    @property
    def nb_flips(self):
        n = self._nb_flips
        self._nb_flips = 0
        return n

    def step(self):
        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get grad and momentum buffer
                grad = p.grad.data
                momentum_buffer = self.state[p]['momentum_buffer']
                
                # Assert that gradient values are integers
                assert torch.all(torch.eq(grad, torch.round(grad))), "Gradients must contain only integer values"
                # Assert that weights are binary (0 or 1)
                assert torch.all(torch.logical_or(p.data == 0, p.data == 1)), "Weights must contain only binary values (0 or 1)"
                
                # Update momentum buffer
                if momentum > 0:
                    momentum_buffer = momentum * momentum_buffer + (1 - dampening) * grad
                else:
                    momentum_buffer = grad
                
                self.state[p]['momentum_buffer'] = momentum_buffer
                
                # Determine which weights to flip based on momentum buffer and threshold
                param_to_flip = (p.data == 1) * (momentum_buffer > self.thresh) + \
                                (p.data == 0) * (momentum_buffer < -self.thresh)
                
                # Log stats for debugging
                get_tensor_stats(momentum_buffer)
                
                # Flip selected weights
                p.data[param_to_flip] = torch.logical_not(p.data[param_to_flip]).float()
                
                # Count flips
                self._nb_flips += float(param_to_flip.float().sum())
                
                # Reset momentum for flipped weights (optional, but can help stabilize training)
                momentum_buffer[param_to_flip] = 0.0

class BoldProbabilisticOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr: float, thresh: int):
        """
        Probabilistic optimizer for boolean networks with linear probability ramp.
        
        Args:
            params: Model parameters
            lr: Learning rate (for API compatibility)
            thresh: Base threshold value - gradients beyond this have 100% flip probability
        """
        defaults = dict(lr=lr, thresh=thresh)
        super(BoldProbabilisticOptimizer, self).__init__(params, defaults)
        self._nb_flips = 0
    
    @property
    def nb_flips(self):
        n = self._nb_flips
        self._nb_flips = 0
        return n
    
    def _linear_prob(self, x, threshold):
        """
        Linear probability function - probability increases linearly from 0 to 1
        as x approaches the threshold.
        
        Args:
            x: Input value (gradient)
            threshold: Threshold value where probability reaches 1.0
            
        Returns:
            Probability between 0 and 1, values >= threshold have probability 1.0
        """
        # Clamp between 0 and 1
        return torch.clamp(x / threshold, 0, 1)
    
    def step(self):
        for group in self.param_groups:
            thresh = group['thresh']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient
                grad = p.grad.data
                
                # Assert conditions for boolean networks
                assert torch.all(torch.eq(grad, torch.round(grad))), "Gradients must contain only integer values"
                assert torch.all(torch.logical_or(p.data == 0, p.data == 1)), "Weights must contain only binary values (0 or 1)"
                
                # Calculate flip probabilities with linear ramp
                # For weights of 1, probability increases linearly with gradient value
                flip_prob_weights_1 = self._linear_prob(grad, thresh)
                
                # For weights of 0, probability increases linearly with negative gradient value
                flip_prob_weights_0 = self._linear_prob(-grad, thresh)
                
                # Combine into a single probability tensor based on current weight values
                flip_probs = torch.where(p.data == 1, flip_prob_weights_1, flip_prob_weights_0)
                
                # Generate random values
                random_vals = torch.rand_like(flip_probs)
                
                # Determine which weights to flip based on probability
                param_to_flip = random_vals < flip_probs
                
                # Log statistics for debugging
                get_tensor_stats(grad)
                get_tensor_stats(flip_probs)
                
                # Flip selected weights
                p.data[param_to_flip] = torch.logical_not(p.data[param_to_flip]).float()
                
                # Count flips
                self._nb_flips += float(param_to_flip.float().sum())