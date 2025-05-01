import torch
import torch.nn as nn
import config
from torch import Tensor , autograd
from torch.nn import functional as F

# ---------------------------------- MARK: IntL1Loss (Recommended) ----------------------------------

class IntL1Loss(nn.Module):
    def __init__(self, activation_range: tuple[int, int]):
        super().__init__()
        self.activation_range = activation_range

    def forward(self, X, target):
        return IntL1LossF.apply(X, target, self.activation_range)
    
class IntL1LossF(autograd.Function):
    @staticmethod
    def forward(ctx, X, target, activation_range):
        eff_target = -torch.ones_like(X) * activation_range[1]
        eff_target[torch.arange(X.size(0)), target] = activation_range[1] * config.args.loss_int_l1_salient
        loss = torch.mean(torch.abs(X - eff_target))

        ctx.save_for_backward(X, target, eff_target)
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        X, target, eff_target = ctx.saved_tensors
        # grad_X = eff_target - X
        grad_X = X - eff_target
        return grad_X * grad_output, None, None
    




# ------------------------- MARK: MixtypeXORLoss (Loss value is not meaningful, but good performance) ----------------

class MixtypeXORLoss(nn.Module):
    """
    The XORMismatchLoss above kills some signals.
    Here propose MixtypeXORLoss that directly acts on the logits.
    The gradient produced by this loss is boolean with representation of T and F as 1 and 0.
    When using this loss, the output of the model shall be the logits produced by some bold linear layer,
    Not the output of the boolean-activation

    Types:
        - input.shape = (batch_size, num_classes)
        - target.shape = (batch_size, )

        - input.dtype = int
        - target.dtype = int is the index of the correct class.

        For a single sample, the loss is defined as:
            loss = xor(input_correct_index, T) + sum(xor(input_incorrect_index, F))

            programatically,
            xor(input_corr, T) is -input_corr
            xor(input_incorr, F) is input_incorr
    """
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return MixtypeXORLossF.apply(input, target)

class MixtypeXORLossF(autograd.Function):
    @staticmethod
    def forward(ctx, X, target):
        # X shape: [batch_size, num_classes]
        # target shape: [batch_size]
        
        # Create mask for incorrect classes
        batch_size = X.size(0)
        num_classes = X.size(1)
        mask = torch.ones(batch_size, num_classes, device=X.device)
        mask.scatter_(1, target.unsqueeze(1), 0)  # Set correct class to 0
        ctx.save_for_backward(X, target, mask)
        
        # Compute loss for correct and incorrect classes
        loss_corr = -num_classes * torch.sum(X[torch.arange(batch_size), target])  # Sum correct class logits
        loss_incorr = torch.sum(X * mask)  # Sum incorrect class logits
        
        return loss_corr + loss_incorr

    @staticmethod
    def backward(ctx, grad_output):
        X, target, mask = ctx.saved_tensors
        batch_size = X.size(0)
        num_classes = X.size(1)
        grad_X = torch.ones_like(X)  # All incorrect classes get 1
        if config.args.float16:
            grad_X = grad_X.to(torch.float16)
            grad_output = grad_output.to(torch.float16)

        # Set correct class gradients to -1
        grad_X[torch.arange(batch_size), target] = -num_classes

        # the G_X shall be interpreted as boolean gradients, i.e. the prev bold layer should have bool_backprop=True
        # the interpretation is very straightforward.
        # For correct class index we have dL/dy = not T = F, indeed, when y gets larger, L decreases. 
        # Because False means the direction of change is opposite.
        # For incorrect class index we have dL/dy = T, indeed, when y gets larger, L increases. 
        # because True means the direction of change is the same.
        return grad_X * grad_output, None


    

# ---------------------------------- MARK: BooleanLoss ----------------------------------

class BoolLoss(autograd.Function):
    @staticmethod
    def forward(ctx, pred, target):
        ctx.save_for_backward(pred, target)
        loss = torch.sum(torch.abs(pred - target)).float()
        return loss

    @staticmethod 
    def backward(ctx, grad_output):
        pred, target = ctx.saved_tensors
        grad_pred = torch.logical_not(target) 
        return grad_pred, None

class BooleanLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        return BoolLoss.apply(pred, target)

# ---------------------------------- MARK: XORMismatchLoss ----------------------------------

class XORMismatchLoss(nn.Module):
    """
    For a multi-class classification, the loss count the number of mismatches of the encoding.
    Input:
        output: [batch_size, num_classes]
        target: [batch_size]
            
        the value type for output is boolean, represented by 0 and 1. The output is not necessarily be one-hot encoded.
        which means the output might fail to produce the top-1 prediction. 
        
        target contains the indices of the correct classes.

    Output:
        loss: int. The number of mismatches encoding across the batch.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return XORMismatchLossF.apply(output, target)

class XORMismatchLossF(autograd.Function):
    @staticmethod
    def forward(ctx, X, target):
        ctx.save_for_backward(X, target)
        # Convert target indices to one-hot for comparison
        target_onehot = F.one_hot(target, num_classes=X.size(1)).float()
        loss = torch.sum(~torch.all(X == target_onehot, dim=1)).float()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        X, target = ctx.saved_tensors
        # Convert target indices to one-hot for gradient computation
        target_onehot = F.one_hot(target, num_classes=X.size(1)).float()
        return torch.logical_not(target_onehot) * grad_output, None



# ---------------------------------- MARK: IntScalingLoss ----------------------------------

class IntScalingLoss(nn.Module):
    """
    Input:
        X.shape = (batch_size, num_classes)
        X.dtype = int  (the model logits output)
        target.shape = (batch_size, )
        target.dtype = int (the index of the correct class)

    Forward pass:
        1. Centering to zero. The logits is added by -min(logits). Output X'
        2. scaling everything by alpha. X'' = alpha * X', target' = alpha * target
        3. Integer division X''' = X'' // max(X'). 
        4. Calculate distance(X''', target')

    Backward pass:
        G_X = target' - X'''
    """
    def __init__(self, alpha: int):
        super().__init__()
        self.alpha = alpha

    def forward(self, X, target):
        return IntScalingLossF.apply(X, target, self.alpha)

class IntScalingLossF(autograd.Function):
    @staticmethod
    def forward(ctx, X, target, alpha):
        # Center each sample independently by finding min along dim=1
        X_centered = X - torch.min(X, dim=1, keepdim=True)[0]
        X_scaled = X_centered * alpha
        X_out = X_scaled // torch.max(X_centered, dim=1, keepdim=True)[0]
        if torch.isnan(X_out).any():
            print("NaN values detected in X_out, replacing with 1")
            X_out = torch.where(torch.isnan(X_out), torch.ones_like(X_out), X_out)
        target_onehot = F.one_hot(target, num_classes=X.size(1)).float()
        target_scaled = target_onehot * alpha
        loss = torch.sum(torch.abs(X_out - target_scaled))
        ctx.save_for_backward(X, target, X_out, target_scaled)  # Save X_out in context
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        _, _, X_out, target_scaled = ctx.saved_tensors  # Retrieve X_out from context
        G_X = X_out - target_scaled
        return G_X * grad_output, None, None
