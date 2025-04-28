import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch import Tensor , autograd
from typing import Any , List , Optional , Callable
import config



################### MARK: BoolActvWithThreshDiscrete ###################

def get_spread(X):
    if config.args.spread is not None:
        return config.args.spread
    elif config.args.spread_std:
        return (X.mean() - X).abs().mean()
    elif config.args.spread_fix_ratio is not None:
        # return config.args.fix_ratio_spread * (X.mean() - X).abs().mean()
        assert False, "not implemented"
    else:
        assert False, "cannot resolve spread parameter"

class ActvFunctionWithThreshDiscrete(autograd.Function):
    @staticmethod
    def forward(ctx, X, sup, spread, output_range, id):
        ctx.save_for_backward(X)
        ctx.sup = sup
        ctx.spread = get_spread(X)
        ctx.output_range = output_range
        ctx.id = id 

        # std = torch.std(X)
        # ctx.std = X

        if config.args.float16:
            S = 2 * torch.ge(X,sup // 2).to(torch.float16) - 1.
        else:
            # S = 2 * torch.ge(X,sup // 2).float() - 1.
            # S = torch.clamp(X, output_range[0], output_range[1])
            S = X.clamp(output_range[0], output_range[1])
        return S

    @staticmethod
    def backward(ctx, Z):
        X, = ctx.saved_tensors
        sup = ctx.sup
        spread = ctx.spread

        # print(torch.std(X))

        dist = torch.abs(X - sup // 2)
        # Create a mask where distance is less than spread
        if config.args.float16:
            G_X = torch.zeros_like(dist).to(torch.float16)
            G_X[dist < spread] = 1
        else:
            G_X = torch.zeros_like(dist)
            G_X[dist < spread] = 1

        # Calculate number of zero gradients
        # num_zeros = torch.sum(G_X == 0).item()
        # Calculate total number of gradients
        # total_gradients = G_X.numel()
        # Calculate percentage of zero gradients
        # zero_grad_percentage = num_zeros / total_gradients
        # config.hooks[f'0_grad_{ctx.id}'] = (num_zeros, total_gradients, zero_grad_percentage)

        G_X = Z * G_X        
        return G_X, None, None, None, None
        
class BoolActvWithThreshDiscrete(nn.Module):
    id = 0
    def __init__(self, sup, spread, output_range = (-1,1)):
        super().__init__()
        self.sup = sup
        self.spread = spread
        self.output_range = output_range
        self.id = BoolActvWithThreshDiscrete.id
        BoolActvWithThreshDiscrete.id += 1
    def forward(self, X) :
        return ActvFunctionWithThreshDiscrete.apply(X, self.sup, self.spread, self.output_range, self.id)
    



################### MARK: normal linear layer with -1 and 1 as weights. No bias ###################
# equivalent to XNORLinear

class XNORLinear(nn.Linear):
    def __init__(self, in_features, out_features, bool_bprop: bool = False):
        super().__init__(in_features, out_features, bias=False)
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize weights to either -1.0 or 1.0
        random_values = torch.randint(0, 2, self.weight.shape)
        if config.args.float16:
            self.weight = nn.Parameter(2.0 * random_values.to(torch.float16) - 1.0)
        else:
            self.weight = nn.Parameter(2.0 * random_values.float() - 1.0)



################### MARK: normal conv2d layer with -1 and 1 as weights. No bias ###################
# equivalent to XNORConv2d

class XNORConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weights to either -1.0 or 1.0
        random_values = torch.randint(0, 2, self.weight.shape)
        if config.args.float16:
            self.weight = nn.Parameter(2.0 * random_values.to(torch.float16) - 1.0)
        else:
            self.weight = nn.Parameter(2.0 * random_values.float() - 1.0)


################### MARK: XNORLinear with explicit implementation ###################

class XNORFunctionManual(autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B, bool_bprop: bool):
        ctx.save_for_backward(X, W, B)
        ctx.bool_bprop = bool_bprop

        S = X[:, None, :] * W[None, :, :] 
        S = S.sum(dim=2)
        return S

    @staticmethod
    def backward(ctx, Z):
        if ctx.bool_bprop:
            raise NotImplementedError("Boolean backprop is not implemented for XNORLinear")

        assert torch.all(torch.eq(Z, torch.round(Z))), f"Z must contain only integer values, but got {Z}"
        X, W, _ = ctx.saved_tensors

        G_X = Z.mm(W)
        G_W = Z.t().mm(X)

        return G_X, G_W, None, None
        
class XNORLinearManual(nn.Linear):
    def __init__(self, in_features : int , out_features : int , bool_bprop : bool = False , ** kwargs ):
        super(XNORLinearManual, self).__init__(in_features ,out_features , ** kwargs )
        self.bool_bprop = bool_bprop
  
    def reset_parameters(self):
        # initialize the weights with either 1.0 or -1.0
        random_values = torch.randint(0, 2, self.weight.shape)
        self.weight = nn.Parameter(2 * random_values.float() - 1)
  
        if self.bias is not None:
            self.bias = nn.Parameter(2 * torch.randint(0, 2, (self.out_features,)).float() - 1)
  
    def forward (self, X) :
        return XNORFunctionManual.apply(X, self.weight , self.bias , self.bool_bprop)




################### MARK: XORLinear ###################

def backward_bool(ctx, Z):
    # Assert that Z only contains binary values (0 or 1)
    assert torch.all(torch.logical_or(Z == 0, Z == 1)), "Z must contain only binary values (0 or 1)"
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
    # assert all Z values are integers
    assert torch.all(torch.eq(Z, torch.round(Z))), f"Z must contain only integer values, but got {Z}"
    X, W, B = ctx.saved_tensors

    """
    Boolean variation of input processed using torch avoiding loop:
    -> xor(Z: Real, W: Boolean) = -Z * emb(W)
    -> emb(W): T->1, F->-1 => emb(W) = 2W - 1
    => delta(Loss)/delta(X) = Z*(1-2W)
    """
    # G_X = Z.mm(1 - 2 * W)
    G_X = Z.mm(-W)

    """
    Boolean variation of weights processed using torch avoiding loop:
    -> xor(Z: Real, X: Boolean) = -Z * emb(X)
    -> emb(X): T->1, F->-1 => emb(X) = 2X - 1
    => delta(Loss)/delta(W) = Z^T * (1-2X)
    """
    # G_W = Z.t().mm(1 - 2 * X)
    G_W = Z.t().mm(-X)

    """ Boolean variation of bias """
    if B is not None:
        G_B = Z.sum(dim=0)

    # Return
    return G_X, G_W, None

     
class XORFunction(autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B, bool_bprop: bool):
        ctx.save_for_backward(X, W, B)
        ctx.bool_bprop = bool_bprop

        # Elementwise XOR logic
        # S = torch.logical_xor(X[:, None, :], W[None, :, :])
        S = -X[:, None, :] * W[None, :, :]

        # Sum over the input dimension
        S = S.sum(dim=2) + B * 0.0

        # 0-centered for use with BatchNorm when preferred
        # S = S - W.shape[1] / 2
    
        return S

    @staticmethod
    def backward(ctx, Z):
        if ctx.bool_bprop:
            G_X, G_W, G_B = backward_bool(ctx, Z)
        else:
            G_X, G_W, G_B = backward_real(ctx, Z)

        return G_X, G_W, None, None
        
class XORLinear(nn.Linear):
    # bool_bprop dictates how to interpret the gradient. 
    #    setting True means  interpret 0 and 1 as False and True. 
    #    setting False means interpret 0 and 1 as float number 0 and 1.
    def __init__(self, in_features : int , out_features : int , bool_bprop : bool , ** kwargs ):
        super(XORLinear, self).__init__(in_features ,out_features , ** kwargs )
        self.bool_bprop = bool_bprop
  
    def reset_parameters(self):
        # self.weight = nn.Parameter(torch.randint(0, 2, self.weight.shape).float())#
        self.weight = nn.Parameter(2. * torch.randint(0, 2, self.weight.shape).float() - 1.)#
  
        if self.bias is not None:
            self.bias = nn.Parameter(2 * torch.randint(0, 2, (self.out_features,)).float() - 1)

  
    def forward (self, X) :
        return XORFunction.apply(X, self.weight , self.bias , self.bool_bprop)


################### MARK: ANDLinear ###################

# not working, might be some bug in the implementation?

class ANDFunction(autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B, bool_bprop: bool):
        ctx.save_for_backward(X, W, B)
        ctx.bool_bprop = bool_bprop

        # Element-wise multiplication with sign handling
        abs_X = torch.abs(X)
        abs_W = torch.abs(W)
        sign_X = torch.sign(X)
        sign_W = torch.sign(W)
        
        # Compute the sign of the product: positive if both are positive, negative otherwise
        # Use broadcasting correctly by expanding dimensions
        product_sign = torch.where((sign_X[:, None, :] > 0) & (sign_W[None, :, :] > 0), 
                                 torch.ones_like(abs_X[:, None, :]), 
                                 -torch.ones_like(abs_X[:, None, :]))
        
        # Apply broadcasting and compute the result
        S = (abs_X[:, None, :] * abs_W[None, :, :]) * product_sign

        S = S.sum(dim=2)
        return S

    @staticmethod
    def backward(ctx, Z):
        if ctx.bool_bprop:
            raise NotImplementedError("Boolean backprop is not implemented for XNORLinear")

        assert torch.all(torch.eq(Z, torch.round(Z))), f"Z must contain only integer values, but got {Z}"
        X, W, _ = ctx.saved_tensors

        # Generate W_eff: maps to 1.0 if w==1.0, maps to 0.0 if w==-1.0
        # W_eff = (W + 1) / 2  # Convert from [-1, 1] to [0, 1]
        # G_X = Z.mm(W_eff)
        # G_W = Z.t().mm(X)

        W_eff = torch.where((W >= 0) | ((W < 0) & (X[:, None, :] < 0)), 
                           torch.ones_like(W), 
                           -torch.ones_like(W))
        
        # G_X = Z.mm(W_eff)
        G_X = Z[:, :, None] * W_eff
        G_X = G_X.sum(dim=1)

        # print("X.shape: ", X.shape)
        # print("W.shape: ", W.shape)
        # X.shape:  torch.Size([256, 128])
        # W.shape:  torch.Size([10, 128])
        X_eff = torch.where((X[:, None, :] >= 0) | ((X[:, None, :] < 0) & (W[None, :, :] < 0)),     
                           torch.ones_like(X[:, None, :]), 
                           -torch.ones_like(X[:, None, :]))
        # print("X_eff.shape: ", X_eff.shape)
        # print("Z.shape: ", Z.shape)
        G_W = Z[:, :, None] * X_eff
        G_W = G_W.sum(dim=0)

        # G_W = Z.t().mm(X_eff)

        return G_X, G_W, None, None
        
class ANDLinear(nn.Linear):
    def __init__(self, in_features : int , out_features : int , bool_bprop : bool = False , ** kwargs ):
        super(ANDLinear, self).__init__(in_features ,out_features , ** kwargs )
        self.bool_bprop = bool_bprop
  
    def reset_parameters(self):
        # initialize the weights with either 1.0 or -1.0
        random_values = torch.randint(0, 2, self.weight.shape)
        self.weight = nn.Parameter(2 * random_values.float() - 1)
  
        if self.bias is not None:
            self.bias = nn.Parameter(2 * torch.randint(0, 2, (self.out_features,)).float() - 1)
  
    def forward (self, X) :
        return ANDFunction.apply(X, self.weight , self.bias , self.bool_bprop)

def test_ANDLinear():
    # it seems correct
    layer = ANDLinear(3, 2, bool_bprop=False)
    print("W: \n", layer.weight)
    x = (2 * torch.randint(0, 2, (4, 3)).float() - 1)
    x.requires_grad_(True)  # Enable gradient tracking
    print("X: \n", x)
    output = layer(x)
    print("output: \n", output)
    output = output.sum()
    output.backward()
    print("output: ", output)
    print("X.grad: \n", x.grad)
    print("param grads: \n", layer.weight.grad)





################### MARK: BoolActv ###################

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


################### MARK: Test ###################

def test_mixtype_xor_linear():
    layer = MixtypeXORLinear(10, 5)
    x = torch.randint(0, 2, (32, 10)).float()
    print("x.shape: ", x.shape)
    print("output shape: ", layer(x).shape)
    output = layer(x)
    output = output.sum()
    output.backward()
    print("param grads: ", layer.weight.grad)

if __name__ == "__main__":
    test_ANDLinear()

