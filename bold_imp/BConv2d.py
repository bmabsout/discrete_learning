import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch import Tensor , autograd
from typing import Any , List , Optional , Callable
from vgg import VGG

# Me trying to implement the Conv2d layer with XOR operation as in BOLD
#
# The replacement is simple: instead of multiplying the weights with the input, we XOR them
# The XOR need to consider taking input of type (A, B), where A is any type, and B is boolean (0 or 1)
class XORConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(XORConv2d, self).__init__()
        assert bias == False, "Bias is not supported yet. It's one boolean value anyway, don't think it is a big deal."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        # initialize the weights with either 0 or 1 in float
        binary_values = torch.randint(0, 2, self.weight.shape, dtype=torch.float)
        self.weight.data = binary_values

        if self.bias is not None:
            raise NotImplementedError("Bias is not supported yet. It's one boolean value anyway, don't think it is a big deal.")
            
    def forward(self, x):
        return XOR2DConvFunction.apply(x, self.weight, self.bias)

# parameter basics:
# X: input tensor
#    - shape: (batch_size, in_channels, height, width)
# W: weight tensor
#    - shape: (out_channels, in_channels, kernel_size, kernel_size)
#
# Basic operation
#     Same as conventional 2DConv, but instead of multiplication, we use XOR
#     The XOR is defined to be XOR(A, B) = A * (1 - 2*B)
#     The basic intuition is that, if B is true (i.e. B==1), then XOR(A, B) = -A, 
#         thus when A>0, the net result is negative, and when A<0, the net result is positive
#     If B is false (i.e. B==0), then XOR(A, B) = A, then if A>0, the net result is positive, and if A<0, the net result is negative
class XOR2DConvFunction(autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B):
        ctx.save_for_backward(X, W, B)
        assert B is None, "Bias is not supported yet. It's one boolean value anyway, don't think it is a big deal."

        # To conduct 2DConv, and instead of using multiplication, we use XOR
        # no need for number of input channels, because the ker and input have same number of channels
        batch_size, _ , in_height, in_width = X.shape 
        out_channels, _, kernel_height, kernel_width = W.shape

        # Calculate output dimensions
        padding = 0
        dilation = 1
        stride = 1
        groups = 1
        out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
        out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1
    
        # Initialize output tensor
        output = torch.zeros(batch_size, out_channels, out_height, out_width)
        
        # assuming padding = 0
        input_padded = X

        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_in = h_out * stride
                        w_in = w_out * stride
                        
                        # Extract the patch from input
                        patch = input_padded[b, :, h_in:h_in+kernel_height, w_in:w_in+kernel_width]
                        
                        # for a weight w in {0,1}, we do 1 - 2*w. So T maps to -1, and F maps to 1
                        # this mulitply with the element in patch
                        # TODO: The primitive form in paper, we know the semantic of sum is counting the number of True, given that weight and input are boolean
                        # However, here we have mixed input, X can by Any and W is boolean. The XOR is clearly defined, but the sum we have to speculate to be the normal arithmetic sum. Athough the type can be Any, the float is at most summed but not mulitplied. 
                        # another implication is that given L(A, B) wher A is Some type, and B is boolean, the result is also Some type. 
                        # this raises a new question: whether we should change the activation function: should we still use strict boolean-output activation?
                        output[b, c_out, h_out, w_out] = torch.sum(patch * (1 - 2 * W[c_out]))
    
        S = output
        return S

    @staticmethod
    def backward(ctx, Z):
        G_X, G_W, G_B = backward_real_2DConv(ctx, Z)
        return G_X, G_W, G_B

# Claude provides the template, I modify it to fit the XOR operation
def backward_real_2DConv(ctx, Z):
    # Get the saved tensors from the context
    X, W, B = ctx.saved_tensors
    
    # Retrieve stride and padding from context if they were saved
    stride = ctx.stride if hasattr(ctx, 'stride') else 1
    padding = ctx.padding if hasattr(ctx, 'padding') else 0
    
    # Get shapes
    batch_size, in_channels, in_height, in_width = X.shape
    out_channels, _, kernel_height, kernel_width = W.shape
    
    # Initialize gradients
    grad_X = torch.zeros_like(X)
    grad_W = torch.zeros_like(W)
    grad_B = torch.zeros_like(B) if B is not None else None
    
    # Compute gradient for bias (sum across batch and spatial dimensions)
    if B is not None:
        grad_B = Z.sum(dim=(0, 2, 3))
    
    # Pad the input if needed
    if padding > 0:
        X_padded = torch.nn.functional.pad(X, (padding, padding, padding, padding))
    else:
        X_padded = X
    
    # First get G_W
    # For each example in the batch
    for b in range(batch_size):
        # For each output channel
        for c_out in range(out_channels):
            # For each spatial location in the output
            for h_out in range(Z.shape[2]):
                for w_out in range(Z.shape[3]):
                    # Calculate the corresponding position in the input
                    h_start = h_out * stride
                    w_start = w_out * stride
                    
                    # Get the current gradient value
                    grad_val = Z[b, c_out, h_out, w_out]
                    
                    # Update gradient for the weights
                    for c_in in range(in_channels):
                        for kh in range(kernel_height):
                            for kw in range(kernel_width):
                                h_in = h_start + kh
                                w_in = w_start + kw
                                
                                if padding > 0:
                                    raise NotImplementedError("Padding not supported yet")
                                    if 0 <= h_in < X_padded.shape[2] and 0 <= w_in < X_padded.shape[3]:
                                        grad_W[c_out, c_in, kh, kw] += X_padded[b, c_in, h_in, w_in] * grad_val
                                else:
                                    if 0 <= h_in < in_height and 0 <= w_in < in_width:
                                        # WARNING: here we run into a case that a float-multiplication is inevitable
                                        # i.e. when X and W are both float. This is especially true when the very first BConv2D layer where X is the image

                                        # USE the following if we know X and grad_val are BOTH FLOAT, this comes from the fact XNOR(x,y) is x * y
                                        # the specific term is XNOR(Z, d xor(x,w) / dw) = XNOR(Z, -x) = -x * Z 
                                        # some comments: how many terms eventually aggregated to grad_w[c_out, c_in, kh, kw]? it is Z.shape[2] * Z.shape[3]
                                        grad_W[c_out, c_in, kh, kw] += -X[b, c_in, h_in, w_in] * grad_val

                                        # USE the following if we know X is BOOL and the representaiton is F == 0 , T == 1. 
                                        # the specific term is XNOR(Z, d xor(x,w) / dw) = XNOR(Z, not x)
                                        # sanity check:
                                        #   if x --> T, and Z > 0, then the result should be -Z
                                        #          (1 - 2 * 1) * Z = -Z
                                        #   if x --> F, and Z > 0, then the result should be Z
                                        #          (1 - 2 * 0) * Z = Z
                                        # grad_W[c_out, c_in, kh, kw] += (1 - 2 * X[b, c_in, h_in, w_in]) * grad_val

    # For each example in the batch
    for batch_index in range(batch_size):
        # For each input channel
        for c_in in range(in_channels):
            # For each spatial location in the input
            for h_in in range(in_height):
                for w_in in range(in_width):
                    # For each output channel
                    for c_out in range(out_channels):
                        # For each position in the filter
                        Z_idx_set = gather_relevant_gradients(Z, batch_index, c_out, h_in, w_in, kernel_height, kernel_width)

                        for idh,idw in Z_idx_set:
                            # the associated weight of the gradient at (a,b)
                            w = W[c_out, c_in, kernel_height - (h_in - idh) - 1, kernel_width - (w_in - idw) - 1]
                            grad_X[batch_index, c_in, h_in, w_in] += (1 - 2 * w) * Z[batch_index, c_out, idh, idw]
                        
    return grad_X, grad_W, grad_B

def gather_relevant_gradients(Z, batch_idx, c_out, h_in, w_in, KH, KW):
    """
    Given the batch index, output channel index, and input position,
    gather all positions in the output gradient Z that involve this input position.
    """
    res_idx = []
    # Determine output dimensions
    out_height, out_width = Z.shape[2], Z.shape[3]
    
    # Calculate the range of output positions that use the input at (h_in, w_in)
    h_start = max(0, h_in - KH + 1)
    h_end = min(h_in + 1, out_height)
    w_start = max(0, w_in - KW + 1)
    w_end = min(w_in + 1, out_width)
    
    # Collect all relevant output positions
    for h_out in range(h_start, h_end):
        for w_out in range(w_start, w_end):
            res_idx.append((h_out, w_out))
            
    return res_idx

def test_forward():
    # Test the forward pass of the XORConv2d layer
    # (in_channels, out_channels, kernel_size)
    xor_conv = XORConv2d(2, 1, 3)
    print('W:')
    print(xor_conv.weight)

    # Create a random input
    # input = torch.randn(1, 3, 5, 5)
    input = torch.randint(low=0, high=10, size=(1, 2, 5, 5))
    print('input')
    print(input)
    output = xor_conv(input)
    print('output')
    print(output)

# some local test
if __name__ == "__main__":
    # Test the XORConv2d layer
    # xor_conv = XORConv2d(3, 4, 3)
    # print(xor_conv.weight)
    # print(xor_conv.weight.shape)

    test_forward()