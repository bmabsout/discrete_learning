import torch
import torch.nn as nn
from torch import Tensor , autograd

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
        ouput: [batch_size, num_classes]
        target: [batch_size, num_classes]
            
        the value type for both is boolean, represented by 0 and 1. The output is not necessarily be one-hot encoded.
        which means the output might fail to produce the top-1 prediction. 
        
        The target requires to be one-hot encoded.

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
        loss = torch.sum(~torch.all(X == target, dim=1)).float()
        return loss


    @staticmethod
    def backward(ctx, Z):
        _, target = ctx.saved_tensors
        return torch.logical_not(target), None

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
        ctx.save_for_backward(X, target)
        # X shape: [batch_size, num_classes]
        # target shape: [batch_size]
        
        # Create mask for incorrect classes
        batch_size = X.size(0)
        num_classes = X.size(1)
        mask = torch.ones(batch_size, num_classes, device=X.device)
        mask.scatter_(1, target.unsqueeze(1), 0)  # Set correct class to 0
        
        # Compute loss for correct and incorrect classes
        loss_corr = -torch.sum(X[torch.arange(batch_size), target])  # Sum correct class logits
        loss_incorr = torch.sum(X * mask)  # Sum incorrect class logits
        
        return loss_corr + loss_incorr

    @staticmethod
    def backward(ctx, grad_output):
        X, target = ctx.saved_tensors
        batch_size = X.size(0)
        num_classes = X.size(1)
        
        # Initialize gradient tensor
        grad_X = torch.ones_like(X)  # All incorrect classes get 1 (the representation of T)
        
        # Set correct class gradients to 0 (the representation of F)
        grad_X[torch.arange(batch_size), target] = 0

        # the G_X shall be interpreted as boolean gradients, i.e. the prev bold layer should have bool_backprop=True
        # the interpretation is very straightforward.
        # For correct class index we have dL/dy = not T = F, indeed, when y gets larger, L decreases. 
        # Because False means the direction of change is opposite.
        # For incorrect class index we have dL/dy = T, indeed, when y gets larger, L increases. 
        # because True means the direction of change is the same.
        return grad_X * grad_output, None

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
    test_mixtype_xor_loss()