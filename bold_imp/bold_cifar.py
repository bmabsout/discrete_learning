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
        S = S - W.shape[1] / 2
        
        # output is not boolean???????

        return S

    @staticmethod
    def backward(ctx, Z):
        if ctx.bool_bprop:
            G_X, G_W, G_B = backward_bool(ctx, Z)
        else:
            G_X, G_W, G_B = backward_real(ctx, Z)

        return G_X, G_W, G_B, None
        
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
        
class XORLinear(nn.Linear):
    def __init__(self, in_features : int , out_features : int , bool_bprop : bool , ** kwargs ):
        super(XORLinear, self). __init__(in_features ,out_features , ** kwargs )
        self.bool_bprop = bool_bprop
  
    def reset_parameters(self):
        self.weight = nn.Parameter(torch.randint(0, 2, self.weight.shape).float())#
  
        if self.bias is not None:
          self.bias = nn.Parameter(torch.randint(0, 2, (self.out_features,)).float())
  
    def forward (self, X) :
        return XORFunction.apply(X, self.weight , self.bias , self.bool_bprop)

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
        
        accum = param_group['ratios'][idx] * param_group['accums'][idx] + param_group['lr'] * param.grad.data
        param_group['accums'][idx] = accum
        #print(param.grad.data.mean(),accum.mean())
        param_to_flip = accum * (2 * param.data - 1) >= 1
        param.data[param_to_flip] = torch.logical_not(param.data[param_to_flip]).float()
        param_group['accums'][idx][param_to_flip] = 0.
        param_group['ratios'][idx] = 1 - param_to_flip.float().mean()
        self._nb_flips += float(param_to_flip.float().sum())


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.init_conv = nn.Conv2d(3,1,kernel_size=5)
        self.bool_fc1 = XORLinear(28*28, 128,bool_bprop=False)
        self.bool_fc2 = XORLinear(128, 64,bool_bprop=False)
        self.bool_fc3 = XORLinear(64, 10,bool_bprop=False)
        self.actv = BoolActv()
        self.final_fc = nn.Linear(10,10)

    def forward(self, x):
        x=self.init_conv(x)
        # Active the float value to boolean
        # x=self.actv(x)
        
        x = x.reshape(-1,28*28)
        x = self.bool_fc1(x)
        
        # Active the interger value to boolean
        x = self.actv(x)
        
        x = self.bool_fc2(x)
        x = self.actv(x)
        x = self.bool_fc3(x)
        
        # x = self.final_fc(x)
        
        output = F.log_softmax(x, dim=1)
        return output


class VanillaNet(nn.Module):
    def __init__(self):
        super(VanillaNet, self).__init__()
        self.init_conv = nn.Conv2d(3,6,kernel_size=5)
        self.conv1 = nn.Conv2d(6,1,kernel_size=5)
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.final_fc = nn.Linear(10,10)

    def forward(self, x):
        print(x.shape)
        x=self.init_conv(x)
        print(x.shape)
        # Active the float value to boolean
        # apply maxpooling
        # x = F.max_pool2d(x, 2)
        x=self.conv1(x)
        print(x.shape)
        asdf
        
        x = x.reshape(-1,28*28)
        x = self.fc1(x)
        
        # Active the interger value to boolean
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        # x = self.final_fc(x)
        
        output = F.log_softmax(x, dim=1)
        return output
        
# Baseline Network
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.init_conv = nn.Conv2d(3,1,kernel_size=5)
#         self.fc1 = nn.Linear(28*28, 128)
#         self.fc2 = nn.Linear(128, 10)
#         self.actv = nn.ReLU()
#         self.final_fc = nn.Linear(10,10)

#     def forward(self, x):
#         x=self.init_conv(x)
        
#         # Active the float value to boolean
#         x=self.actv(x)
        
#         x = x.reshape(-1,28*28)
#         x = self.fc1(x)
        
#         # Active the interger value to boolean
#         x = self.actv(x)
        
#         x = self.fc2(x)
        
#         x = self.final_fc(x)
        
#         output = F.log_softmax(x, dim=1)
#         return output


def train(args, model, device, train_loader, optimizers, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for optimizer in optimizers:
              optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        for optimizer in optimizers:
             optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor()
        ])
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.CIFAR10('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # model = Net().to(device)
    # model = VanillaNet().to(device)
    model = VGG('VGG11').to(device)
    
    
    optimizer = optim.Adam([x for name,x in model.named_parameters() if 'bool_' not in name], lr=args.lr)
    all_optimizers = [optimizer]
    bool_params = [x for name,x in model.named_parameters() if 'bool_' in name]
    if len(bool_params):
         optimizer_bool=BooleanOptimizer(bool_params, lr=args.lr*10)
         all_optimizers.append(optimizer_bool)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, all_optimizers, epoch)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "cifar_cnn.pt")


if __name__ == '__main__':
    main()