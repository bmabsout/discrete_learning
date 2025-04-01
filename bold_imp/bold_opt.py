import torch
from torch import Tensor
from utils import get_tensor_stats


def calculate_flip_probabilities(x, flip_ratio=0.001):
    """
    Scale flip probabilities to achieve the desired flip ratio.
    
    This function calculates probabilities proportional to momentum values,
    scaled so that the expected number of flips equals the target ratio.
    
    Args:
        x: Input tensor of momentum values
        flip_ratio: Target percentage of parameters to flip (default: 0.001)
            
    Returns:
        Probability tensor between 0 and 1, scaled to achieve target flip ratio
    """
    # Take the positive part of x
    x_pos = torch.maximum(torch.zeros_like(x), x)
    
    # Calculate mean of positive values (with safeguard against division by zero)
    avg = torch.mean(x_pos)
    if avg == 0:
        return torch.zeros_like(x)
        
    # Scale probabilities proportionally to positive momentum values
    # and target flip ratio
    return torch.clamp(flip_ratio * x_pos / avg, 0, 1)


def check_param_values(param):
    """
    Validate that parameters meet requirements for boolean optimizers
    
    Args:
        param: Parameter tensor to validate
    """
    # Assert that gradient values are integers
    assert torch.all(torch.eq(param.grad.data, torch.round(param.grad.data))), "Gradients must contain only integer values"
    # Assert that weights are binary (0 or 1)
    assert torch.all(torch.logical_or(param.data == 0, param.data == 1)), "Weights must contain only binary values (0 or 1)"


def update_momentum_buffer(grad, momentum_buffer, momentum, dampening):
    """
    Update momentum buffer with new gradient values
    
    Args:
        grad: Current gradient values
        momentum_buffer: Existing momentum buffer
        momentum: Momentum factor
        dampening: Dampening factor
    Returns:
        Updated momentum buffer
    """
    if momentum > 0:
        return momentum * momentum_buffer + (1 - dampening) * grad
    else:
        return grad


def determine_flips_vanilla(param_data, grad_data, thresh):
    """
    Determine which weights to flip using the vanilla threshold method.
    
    Args:
        param_data: Current parameter values (0 or 1)
        grad_data: Gradient values
        thresh: Threshold for flipping
        
    Returns:
        Boolean mask indicating which weights to flip
    """
    return (param_data == 1) * (grad_data > thresh) + \
           (param_data == 0) * (grad_data < -thresh)


def determine_flips_probabilistic(param_data, signal, flip_ratio):
    """
    Determine which weights to flip using probabilistic method.
    
    Args:
        param_data: Current parameter values (0 or 1)
        signal: Signal for determining flip probabilities (gradient or momentum)
        flip_ratio: Target percentage of parameters to flip
        
    Returns:
        Boolean mask indicating which weights to flip
    """
    # Calculate flip probabilities for weights of 1 and 0
    flip_prob_weights_1 = calculate_flip_probabilities(signal, flip_ratio)
    flip_prob_weights_0 = calculate_flip_probabilities(-signal, flip_ratio)
    
    # Combine into a single probability tensor based on current weight values
    flip_probs = torch.where(param_data == 1, flip_prob_weights_1, flip_prob_weights_0)
    
    # Generate random values
    random_vals = torch.rand_like(flip_probs)
    
    # Determine which weights to flip based on probability
    return random_vals < flip_probs


def format_flip_stats(flips, total, target_ratio=None):
    """
    Format flip statistics for logging
    
    Args:
        flips: Number of flips
        total: Total number of parameters
        target_ratio: Target flip ratio (optional)
        
    Returns:
        Formatted string with statistics
    """
    actual_ratio = flips / total if total > 0 else 0
    
    if target_ratio is not None:
        return f"  Flip stats: {flips}/{total} parameters ({actual_ratio*100:.4f}%), target was {target_ratio*100:.4f}%"
    else:
        return f"  Flip stats: {flips}/{total} parameters ({actual_ratio*100:.4f}%)"


# Base class for all Boolean optimizers
class BaseBooleanOptimizer(torch.optim.Optimizer):
    def __init__(self, params, defaults):
        super(BaseBooleanOptimizer, self).__init__(params, defaults)
        self._nb_flips = 0
        self._total_flips = 0
        self._total_params = 0
    
    @property
    def nb_flips(self):
        n = self._nb_flips
        self._nb_flips = 0
        return n
    
    def step(self):
        """
        Performs a single optimization step.
        This should be overridden by subclasses.
        """
        raise NotImplementedError
    
    def _flip_weights(self, param, param_to_flip):
        """
        Flips the selected weights and updates flip counters
        """
        param.data[param_to_flip] = torch.logical_not(param.data[param_to_flip]).float()
        
        num_flips = param_to_flip.sum().item()
        self._nb_flips += num_flips
        self._total_flips += num_flips
        self._total_params += param.data.numel()
    
    def log_stats(self):
        """
        Log statistics about the flip ratio and reset counters.
        """
        if self._total_params == 0:
            return
            
        flip_ratio = getattr(self, 'flip_ratio', None)
        print(format_flip_stats(self._total_flips, self._total_params, flip_ratio), end="")
              
        # Reset counters after logging
        self._total_flips = 0
        self._total_params = 0


# Legacy implementation, kept for backwards compatibility
class BooleanOptimizer(BaseBooleanOptimizer):
    def __init__(self, params, lr: float):
        super(BooleanOptimizer, self).__init__(params, dict(lr=lr))
        for param_group in self.param_groups:
            param_group['accums'] = [torch.zeros_like(p.data) for p in param_group['params']]
            param_group['ratios'] = [0 for p in param_group['params']]

    def step(self):
        for param_group in self.param_groups:
            for idx, p in enumerate(param_group['params']):
                self.update(p, param_group, idx)

    def update(self, param: Tensor, param_group: dict, idx: int):
        # Assert that gradient values are integers
        assert torch.all(torch.eq(param.grad.data, torch.round(param.grad.data))), "Gradients must contain only integer values"
        accum = param_group['ratios'][idx] * param_group['accums'][idx] + param_group['lr'] * param.grad.data
        param_group['accums'][idx] = accum
        
        param_to_flip = accum * (2 * param.data - 1) >= 1
        param.data[param_to_flip] = torch.logical_not(param.data[param_to_flip]).float()
        param_group['accums'][idx][param_to_flip] = 0.
        param_group['ratios'][idx] = 1 - param_to_flip.float().mean()
        self._nb_flips += float(param_to_flip.float().sum())


class BoldVanillaOptimizer(BaseBooleanOptimizer):
    def __init__(self, params, lr: float, thresh: int):
        defaults = dict(lr=lr, thresh=thresh)
        super(BoldVanillaOptimizer, self).__init__(params, defaults)
        self.thresh = thresh

    def step(self):
        for group in self.param_groups:
            thresh = group.get('thresh', self.thresh)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Validate parameter values
                check_param_values(p)
                
                # Determine which weights to flip based on threshold
                param_to_flip = determine_flips_vanilla(p.data, p.grad.data, thresh)
                
                # Log stats for debugging
                get_tensor_stats(p.grad.data)
                
                # Flip selected weights
                self._flip_weights(p, param_to_flip)


class BoldVanillaMomentumOptimizer(BaseBooleanOptimizer):
    def __init__(self, params, lr: float, thresh: int, momentum: float = 0.9, dampening: float = 0.0):
        defaults = dict(lr=lr, thresh=thresh, momentum=momentum, dampening=dampening)
        super(BoldVanillaMomentumOptimizer, self).__init__(params, defaults)
        self.thresh = thresh
        self.momentum = momentum
        self.dampening = dampening
        
        # Initialize momentum buffers for each parameter
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

    def step(self):
        for group in self.param_groups:
            momentum = group.get('momentum', self.momentum)
            dampening = group.get('dampening', self.dampening)
            thresh = group.get('thresh', self.thresh)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get grad and momentum buffer
                grad = p.grad.data
                momentum_buffer = self.state[p]['momentum_buffer']
                
                # Validate parameter values
                check_param_values(p)
                
                # Update momentum buffer
                momentum_buffer = update_momentum_buffer(grad, momentum_buffer, momentum, dampening)
                self.state[p]['momentum_buffer'] = momentum_buffer
                
                # Determine which weights to flip based on momentum buffer and threshold
                param_to_flip = determine_flips_vanilla(p.data, momentum_buffer, thresh)
                
                # Log stats for debugging
                get_tensor_stats(momentum_buffer)
                
                # Flip selected weights
                self._flip_weights(p, param_to_flip)
                
                # Reset momentum for flipped weights
                momentum_buffer[param_to_flip] = 0.0


class BoldProbabilisticOptimizer(BaseBooleanOptimizer):
    def __init__(self, params, lr: float, flip_ratio: float = 0.001):
        """
        Probabilistic optimizer for boolean networks.
        
        Args:
            params: Model parameters
            lr: Learning rate (for API compatibility)
            flip_ratio: Target percentage of parameters to flip (default: 0.001, i.e., 0.1%)
        """
            
        defaults = dict(lr=lr, flip_ratio=flip_ratio)
        super(BoldProbabilisticOptimizer, self).__init__(params, defaults)
        self.flip_ratio = flip_ratio
    
    def step(self):
        for group in self.param_groups:
            flip_ratio = group.get('flip_ratio', self.flip_ratio)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient
                grad = p.grad.data
                
                # Validate parameter values
                check_param_values(p)
                
                # Determine which weights to flip based on probability
                param_to_flip = determine_flips_probabilistic(p.data, grad, flip_ratio)
                
                # Log statistics for debugging
                get_tensor_stats(grad)
                
                # Flip selected weights
                self._flip_weights(p, param_to_flip)


class BoldProbabilisticMomentumOptimizer(BaseBooleanOptimizer):
    def __init__(self, params, lr: float = None, momentum: float = 0.9, dampening: float = 0.0, flip_ratio: float = 0.001):
        """
        Probabilistic optimizer with momentum for boolean networks.
        
        Args:
            params: Model parameters
            lr: Learning rate (for API compatibility, not used)
            momentum: Momentum factor (default: 0.9)
            dampening: Dampening factor for momentum (default: 0.0)
            flip_ratio: Target percentage of parameters to flip (default: 0.1%)
        """
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, flip_ratio=flip_ratio)
        super(BoldProbabilisticMomentumOptimizer, self).__init__(params, defaults)
        self.momentum = momentum
        self.dampening = dampening
        self.flip_ratio = flip_ratio
        
        # Initialize momentum buffers for each parameter
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['momentum_buffer'] = torch.zeros_like(p.data)
    
    def step(self):
        for group in self.param_groups:
            momentum = group.get('momentum', self.momentum)
            dampening = group.get('dampening', self.dampening)
            flip_ratio = group.get('flip_ratio', self.flip_ratio)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get grad and momentum buffer
                grad = p.grad.data
                momentum_buffer = self.state[p]['momentum_buffer']
                
                # Validate parameter values
                check_param_values(p)
                
                # Update momentum buffer
                momentum_buffer = update_momentum_buffer(grad, momentum_buffer, momentum, dampening)
                self.state[p]['momentum_buffer'] = momentum_buffer
                
                # Determine which weights to flip based on probability
                param_to_flip = determine_flips_probabilistic(p.data, momentum_buffer, flip_ratio)
                
                # Log statistics for debugging
                get_tensor_stats(momentum_buffer)
                
                # Flip selected weights
                self._flip_weights(p, param_to_flip)
                
                # Reset momentum for flipped weights to prevent oscillation
                momentum_buffer[param_to_flip] = 0.0