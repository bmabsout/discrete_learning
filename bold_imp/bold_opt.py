import torch
import config
from torch import Tensor
from typing import Callable, Dict, Any, Protocol, List, Union, Optional
from utils import get_tensor_stats


# Type definitions
class OptimizerState(Dict[str, Any]): pass
class FlipDecider(Protocol):
    """Selects which weights to flip"""
    def __call__(self, weights: Tensor, accumulated_grad: Tensor) -> Tensor: ...

class GradientAccumulator(Protocol):
    """Accumulates gradient information to make flip decisions"""
    def __call__(self, param: Tensor, grad: Tensor, state: OptimizerState) -> Tensor: ...


def calculate_flip_probabilities(gradient_values: Tensor, flip_ratio: float = 0.001) -> Tensor:
    # positive_values = torch.maximum(torch.zeros_like(gradient_values), gradient_values)
    # positive_values = torch.clamp(gradient_values, 0)
    # mean_positive = torch.mean(positive_values)
    # if mean_positive == 0:
    #     return torch.zeros_like(gradient_values)
    # return torch.clamp(flip_ratio * positive_values / mean_positive, 0, 1)

    # mem eff
    mean_positive = torch.mean(gradient_values[gradient_values>0])
    if mean_positive == 0:
        return torch.zeros_like(gradient_values)
    return (flip_ratio * gradient_values / mean_positive).clamp(0,1)

def validate_tensors_for_bool_opt(validate_me: Tensor) -> None:
    # assert torch.all(torch.eq(validate_me.grad.data, torch.round(validate_me.grad.data))), "Gradients must contain only integer values"
    # assert torch.all(torch.logical_or(validate_me.data == 0, validate_me.data == 1)), "Weights must contain only binary values (0 or 1)"
    return None


def update_momentum_buffer(grad: Tensor, momentum_buffer: Tensor, momentum: float, dampening: float) -> Tensor:
    if momentum > 0:
        return momentum * momentum_buffer + (1 - dampening) * grad
    else:
        return grad


# Gradient accumulators
def raw_gradient_accumulator(param: Tensor, grad: Tensor, state: OptimizerState) -> Tensor:
    return grad.data


def get_momentum_accumulator(momentum: float = 0.9, dampening: float = 0.0) -> GradientAccumulator:
    def momentum_accumulator(param: Tensor, grad: Tensor, state: OptimizerState) -> Tensor:
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(param.data)
        
        state['momentum_buffer'] = update_momentum_buffer(
            grad.data, state['momentum_buffer'], momentum, dampening
        )
        
        return state['momentum_buffer']
    
    return momentum_accumulator


# Flip deciders
def get_threshold_decider(thresh: int) -> FlipDecider:
    def threshold_flip_decider(weights: Tensor, accumulated_grad: Tensor) -> Tensor:
        return (weights == 1.0) * (accumulated_grad > thresh) + \
               (weights == 0) * (accumulated_grad < -thresh)
            #    (weights == -1.0) * (accumulated_grad < -thresh)
    
    threshold_flip_decider.thresh = thresh  # type: ignore
    return threshold_flip_decider


def get_probabilistic_decider(flip_ratio: float = 0.001) -> FlipDecider:
    def probabilistic_flip_decider(weights: Tensor, accumulated_grad: Tensor) -> Tensor:
        num_correct, num_total = config.hooks['cur_acc']
        epoch = config.hooks['epoch']

        # per-batch accuracy feedbcak loop
        acc = num_correct / num_total
        k = 1 - (0.99 * acc)

        # flip-ratio decay
        r = epoch // 10
        cur_flip_ratio_decay = config.args.flip_ratio_decay ** r

        flip_prob_weights_1 = calculate_flip_probabilities(accumulated_grad, k * flip_ratio * cur_flip_ratio_decay)
        flip_prob_weights_0 = calculate_flip_probabilities(-accumulated_grad, k * flip_ratio * cur_flip_ratio_decay)
        flip_probs = torch.where(weights == 1.0, flip_prob_weights_1, flip_prob_weights_0)
        random_values = torch.rand_like(flip_probs)

        # if str(weights.shape) == "torch.Size([1024, 784])":
        #     one_randomfloat = torch.rand_like(torch.tensor(1).float())
        #     config.randscaler = one_randomfloat
        # else:
        #     one_randomfloat = config.randscaler

        # random_values = torch.ones_like(flip_probs)
        # random_values = one_randomfloat * torch.ones_like(flip_probs)

        return random_values < flip_probs
    
    probabilistic_flip_decider.flip_ratio = flip_ratio  # type: ignore
    return probabilistic_flip_decider


# Base optimizer class
class BaseBooleanOptimizer(torch.optim.Optimizer):
    def __init__(self, params, defaults, gradient_accumulator: GradientAccumulator, flip_decider: FlipDecider):
        super(BaseBooleanOptimizer, self).__init__(params, defaults)
        self._nb_flips = 0
        self._total_flips = 0
        self._total_params = 0
        self.gradient_accumulator = gradient_accumulator
        self.flip_decider = flip_decider
    
    @property
    def nb_flips(self):
        n = self._nb_flips
        self._nb_flips = 0
        return n
    
    def _flip_weights(self, param: Tensor, weights_to_flip: Tensor):
        param.data[weights_to_flip] = -param.data[weights_to_flip]
        
        num_flips = weights_to_flip.sum().item()
        self._nb_flips += num_flips
        self._total_flips += num_flips
        self._total_params += param.data.numel()
    
    def _reset_momentum_after_flip(self, weights_to_flip: Tensor, state: OptimizerState):
        if 'momentum_buffer' in state:
            state['momentum_buffer'][weights_to_flip] = 0.0
    
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                validate_tensors_for_bool_opt(p)
                
                accumulated_grad = self.gradient_accumulator(p, p.grad, state)
                get_tensor_stats(accumulated_grad)
                
                weights_to_flip = self.flip_decider(p.data, accumulated_grad)
                self._flip_weights(p, weights_to_flip)
                self._reset_momentum_after_flip(weights_to_flip, state)
    
    def log_stats(self):
        if self._total_params == 0:
            return
        
        actual_ratio = self._total_flips / self._total_params
        target_ratio = getattr(self.flip_decider, 'flip_ratio', None)
        
        if target_ratio is not None:
            print(f"  Flip stats: {self._total_flips}/{self._total_params} parameters "
                  f"({actual_ratio*100:.4f}%), target was {target_ratio*100:.4f}%", end="")
        else:
            print(f"  Flip stats: {self._total_flips}/{self._total_params} parameters "
                  f"({actual_ratio*100:.4f}%)", end="")
        
        self._total_flips = 0
        self._total_params = 0


# Factory functions for creating optimizers
def create_vanilla_optimizer(params, lr: float, thresh: int) -> BaseBooleanOptimizer:
    """Creates a vanilla threshold-based optimizer"""
    return BaseBooleanOptimizer(
        params,
        defaults=dict(lr=lr, thresh=thresh),
        gradient_accumulator=raw_gradient_accumulator,
        flip_decider=get_threshold_decider(thresh)
    )


def create_momentum_optimizer(
    params, lr: float, thresh: int, momentum: float = 0.9, dampening: float = 0.0
) -> BaseBooleanOptimizer:
    """Creates a momentum-based threshold optimizer"""
    return BaseBooleanOptimizer(
        params,
        defaults=dict(lr=lr, thresh=thresh, momentum=momentum, dampening=dampening),
        gradient_accumulator=get_momentum_accumulator(momentum, dampening),
        flip_decider=get_threshold_decider(thresh)
    )


def create_probabilistic_optimizer(params, lr: float, flip_ratio: float = 0.001) -> BaseBooleanOptimizer:
    """Creates a probabilistic optimizer"""
    return BaseBooleanOptimizer(
        params,
        defaults=dict(lr=lr, flip_ratio=flip_ratio),
        gradient_accumulator=raw_gradient_accumulator,
        flip_decider=get_probabilistic_decider(flip_ratio)
    )


def create_probabilistic_momentum_optimizer(
    params, lr: Optional[float] = None, momentum: float = 0.9, 
    dampening: float = 0.0, flip_ratio: float = 0.001
) -> BaseBooleanOptimizer:
    """Creates a probabilistic optimizer with momentum"""
    return BaseBooleanOptimizer(
        params,
        defaults=dict(lr=lr, momentum=momentum, dampening=dampening, flip_ratio=flip_ratio),
        gradient_accumulator=get_momentum_accumulator(momentum, dampening),
        flip_decider=get_probabilistic_decider(flip_ratio)
    )