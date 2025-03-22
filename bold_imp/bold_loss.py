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
    test_xor_mismatch_loss()