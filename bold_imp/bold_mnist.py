import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch import Tensor , autograd
from typing import Any , List , Optional , Callable
from BConv2d import XORConv2d, BoolActvWithThresh, BoolActvWithThreshDiscrete
from bold_layers import XORLinear, BoolActv
from bold_opt import BooleanOptimizer


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bool_fc1 = XORLinear(28*28, 64,bool_bprop=False)
        self.actv1 = BoolActvWithThresh(28*28)
        self.bool_fc2 = XORLinear(64, 10,bool_bprop=False)
        self.final_fc = nn.Linear(10,10)

    def forward(self, x):
        x = x.reshape(-1,28*28)
        x = self.bool_fc1(x)
        x = self.actv1(x)
        x = self.bool_fc2(x)
        return F.log_softmax(x, dim=1)
    
class NetThreshold(nn.Module):
    def __init__(self):
        super(NetThreshold, self).__init__()
        self.bool_fc1 = XORLinear(28*28, 64, bool_bprop=False)
        self.actv1 = BoolActvWithThreshDiscrete(28*28, spread=10)
        self.bool_fc2 = XORLinear(64, 2, bool_bprop=False)
        self.final_fc = nn.Linear(1,1)

    def forward(self, x):
        x = x.reshape(-1,28*28)
        x = self.bool_fc1(x)
        x = self.actv1(x)
        x = self.bool_fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, optimizer_bool, epoch):
    model.train()
    total_flips = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data=torch.gt(data,0.5).float()
        
        # Zero gradients for whichever optimizer exists
        if optimizer is not None:
            optimizer.zero_grad()
        if optimizer_bool is not None:
            optimizer_bool.zero_grad()
        
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        
        # Update with whichever optimizer exists
        if optimizer is not None:
            optimizer.step()
        if optimizer_bool is not None:
            optimizer_bool.step()
            # Get the number of flips from the boolean optimizer
            batch_flips = optimizer_bool.nb_flips
            total_flips += batch_flips
        else:
            batch_flips = 0
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tFlips: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), batch_flips))
            if args.dry_run:
                break
    
    print('Total flips in epoch {}: {}'.format(epoch, total_flips))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data=torch.gt(data,0.5).float()
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
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
        print("Using CUDA")
    elif use_mps:
        device = torch.device("mps")
        print("Using MPS")
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
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    
    # Create binary dataset (only digits 0 and 1)
    idx_train = (dataset1.targets == 0) | (dataset1.targets == 1)
    idx_test = (dataset2.targets == 0) | (dataset2.targets == 1)
    
    dataset1.data = dataset1.data[idx_train]
    dataset1.targets = dataset1.targets[idx_train]
    dataset2.data = dataset2.data[idx_test]
    dataset2.targets = dataset2.targets[idx_test]

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    # model = NetThreshold().to(device)
    
    optimizer = optim.Adam([x for name,x in model.named_parameters() if 'bool_' not in name], lr=args.lr)
    optimizer_bool = BooleanOptimizer([x for name,x in model.named_parameters() if 'bool_' in name], lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, optimizer_bool, epoch)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_bnn.pt")


if __name__ == '__main__':
    main()