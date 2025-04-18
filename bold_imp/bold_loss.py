import torch
import torch.nn as nn
import config
from torch import Tensor , autograd
from torch.nn import functional as F

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


        # return torch.ones(1)

    @staticmethod
    def backward(ctx, grad_output):
        X, target, mask = ctx.saved_tensors
        batch_size = X.size(0)
        num_classes = X.size(1)
        # Get the number of instances per class
        # num_instances_per_class = torch.sum(1-mask, dim=0)
        # print(num_instances_per_class)
        # print("sum of X:", torch.sum(X, dim=0))
        # Initialize gradient tensor
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
            

def test_int_scaling_loss():
    # Test case 1
    print("Test case 1:")
    input = torch.tensor([[100., 200., -10., 50.], [-30., -20., 10., 100.]], requires_grad=True)
    target = torch.tensor([0, 1])
    loss = IntScalingLoss(alpha=1000)
    output = loss(input, target)
    print("forward pass logits:")
    print(output)
    output.backward()
    
    print("\nTesting backward pass:")
    print("Input gradients:")
    print(input.grad)



def test_mixtype_xor_loss():
    # Test case 1
    print("Test case 1:")
    # Test backward pass
    input = torch.tensor([[100., 200., -10., 50.], [-30., -20., 10., 100.]], requires_grad=True)
    target = torch.tensor([0, 1])
    loss = MixtypeXORLoss()
    output = loss(input, target)
    print("forward pass logits:")
    print(output)
    output.backward()
    
    print("\nTesting backward pass:")
    print("Input gradients:")
    print(input.grad)
    input.grad = None  # Clear any existing gradients

    # Test case 2
    print("Test case 2:")
    input = torch.tensor([[100., 200., -10., 50.], [-30., -20., 10., 100.]], requires_grad=True)
    target = torch.tensor([1, 0])
    loss = MixtypeXORLoss()
    output = loss(input, target)
    print("forward pass logits:")
    print(output)
    output.backward()
    
    print("\nTesting backward pass:")
    print("Input gradients:")
    print(input.grad)
    

def test_boolean_loss():
    pred = torch.tensor([1, 0, 1, 0])
    target = torch.tensor([1, 1, 0, 0])
    loss = BooleanLoss()
    print(loss(pred, target))

def test_xor_mismatch_loss():
    # Test case 1: Basic mismatch - output has multiple 1s
    output = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    target = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]])  # one-hot encoded
    loss = XORMismatchLoss()
    print("Test case 1 loss:", loss(output, target).item())

    # Test case 2: Perfect match
    output = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]])
    target = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]])  # one-hot encoded
    print("Test case 2 loss:", loss(output, target).item())

    # Test case 3: All mismatches - output has all 1s or all 0s
    output = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]])
    target = torch.tensor([[0, 0, 1, 0], [1, 0, 0, 0]])  # one-hot encoded
    print("Test case 3 loss:", loss(output, target).item())

    # Test case 4: Mixed matches/mismatches with multiple 1s in output
    output = torch.tensor([[1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 0, 0]])
    target = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]])  # one-hot encoded
    print("Test case 4 loss:", loss(output, target).item())

    # Test case 5: Output with no 1s
    output = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
    target = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1]])  # one-hot encoded
    print("Test case 5 loss:", loss(output, target).item())

    # Test case 6: Large batch with varied mismatches
    output = torch.tensor([
        [1, 1, 1, 0],  # multiple 1s
        [0, 0, 0, 0],  # no 1s
        [1, 0, 0, 0],  # matches target
        [0, 1, 1, 0],  # multiple 1s
        [1, 1, 1, 1]   # all 1s
    ])
    target = torch.tensor([
        [0, 0, 0, 1],  # one-hot encoded
        [0, 1, 0, 0],  # one-hot encoded
        [1, 0, 0, 0],  # one-hot encoded
        [0, 0, 1, 0],  # one-hot encoded
        [0, 0, 0, 1]   # one-hot encoded
    ])
    print("Test case 6 loss:", loss(output, target).item())


if __name__ == "__main__":
    # test_xor_mismatch_loss()
    # test_mixtype_xor_loss()
    test_int_scaling_loss()