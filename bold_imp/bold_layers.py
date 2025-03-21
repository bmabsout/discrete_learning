import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch import Tensor , autograd
from typing import Any , List , Optional , Callable
from BConv2d import XORConv2d, BoolActvWithThresh, BoolActvWithThreshDiscrete

def backward_bool(ctx, Z):
    """
    Variation of input:
    - delta(xor(x,w))/delta(x) = neg w
    - delta(Loss)/delta(x) = xnor(z, neg w) = xor(z,w)
    Variation of weights:
    - delta(xor(x,w))/delta(w) = neg x
    - delta(Loss)/delta(x) = xnor(z, neg x) = xor(z,x)
    Variation of bias:
    - bias = xnor(bias, True) ==> Variation of bias is driven in
      the same basis as that of weight with xnor logic and input True.
    Aggregation:
    - Count the number of TRUEs = sum over the Boolean data
    - Aggr = TRUEs - FALSEs = TRUEs - (TOT - TRUEs) = 2 TRUES - TOT
      where TOT is the size of the aggregated dimension
    """
    X, W, B = ctx.saved_tensors

    # Boolean variation of input
    G_X = torch.logical_xor(Z[:, :, None], W[None, :, :])

    # Aggregate over the out_features dimension
    G_X = 2 * G_X.sum(dim=1) - W.shape[0]

    # Boolean variation of weights
    G_W = torch.logical_xor(Z[:, :, None], X[:, None, :])

    # Aggregate over the batch dimension
    G_W = 2 * G_W.sum(dim=0) - X.shape[0]

    # Boolean variation of bias
    if B is not None:
        # Aggregate over the batch dimension
        G_B = 2 * Z.sum(dim=0) - Z.shape[0]

    # Return
    return G_X, G_W, G_B

def backward_real(ctx, Z):
    X, W, B = ctx.saved_tensors

    """
    Boolean variation of input processed using torch avoiding loop:
    -> xor(Z: Real, W: Boolean) = -Z * emb(W)
    -> emb(W): T->1, F->-1 => emb(W) = 2W - 1
    => delta(Loss)/delta(X) = Z*(1-2W)
    """
    G_X = Z.mm(1 - 2 * W)

    """
    Boolean variation of weights processed using torch avoiding loop:
    -> xor(Z: Real, X: Boolean) = -Z * emb(X)
    -> emb(X): T->1, F->-1 => emb(X) = 2X - 1
    => delta(Loss)/delta(W) = Z^T * (1-2X)
    """
    G_W = Z.t().mm(1 - 2 * X)

    """ Boolean variation of bias """
    if B is not None:
        G_B = Z.sum(dim=0)

    # Return
    return G_X, G_W, G_B

     
class XORFunction(autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B, bool_bprop: bool):
        ctx.save_for_backward(X, W, B)
        ctx.bool_bprop = bool_bprop

        # Elementwise XOR logic
        S = torch.logical_xor(X[:, None, :], W[None, :, :])

        # Sum over the input dimension
        S = S.sum(dim=2) + B

        # 0-centered for use with BatchNorm when preferred
        # S = S - W.shape[1] / 2
        
        # output is not boolean???????

        return S

    @staticmethod
    def backward(ctx, Z):
        if ctx.bool_bprop:
            G_X, G_W, G_B = backward_bool(ctx, Z)
        else:
            G_X, G_W, G_B = backward_real(ctx, Z)

        return G_X, G_W, G_B, None
        
class XORLinear(nn.Linear):
    def __init__(self, in_features : int , out_features : int , bool_bprop : bool , ** kwargs ):
        super(XORLinear, self).__init__(in_features ,out_features , ** kwargs )
        self.bool_bprop = bool_bprop
  
    def reset_parameters(self):
        self.weight = nn.Parameter(torch.randint(0, 2, self.weight.shape).float())#
  
        if self.bias is not None:
          self.bias = nn.Parameter(torch.randint(0, 2, (self.out_features,)).float())
  
    def forward (self, X) :
        return XORFunction.apply(X, self.weight , self.bias , self.bool_bprop)



class ActvFunction(autograd.Function):
    @staticmethod
    def forward(ctx, X):
        ctx.save_for_backward(X)

        S = torch.ge(X,0).float()

        return S

    @staticmethod
    def backward(ctx, Z):
        
        dist_thresh = 4
        
        X, = ctx.saved_tensors
        
        # Gradient is defined by the distance to te center
        G_X = torch.maximum(torch.zeros_like(X),dist_thresh-torch.abs(X)).float()
        
        G_X = Z*G_X
        
        return G_X, None
        
class BoolActv(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, X) :
        return ActvFunction.apply(X)