import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from typing import Any , List , Optional , Callable
from utils import get_args, filter_dataset_by_labels

from bold_layers import XORLinear, BooleanLoss, BoolActvWithThreshDiscrete
from bold_opt import BoldVanillaOptimizer

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bool_fc1 = XORLinear(28*28, 64,bool_bprop=False)
        self.actv1 = BoolActvWithThreshDiscrete(28*28, spread=10)
        self.bool_fc2 = XORLinear(64, 1,bool_bprop=True)  
        self.actv2 = BoolActvWithThreshDiscrete(64, spread=10)

    def forward(self, x):
        x = x.reshape(-1,28*28)
        x = self.bool_fc1(x)
        x = self.actv1(x)
        x = self.bool_fc2(x)
        x = self.actv2(x)
        return x



def train(args, model, device, train_loader, optimizer, optimizer_bool, epoch):
    model.train()
    total_flips = 0
    criterion = BooleanLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data=torch.gt(data,0.5).float()
        
        if optimizer is not None:
            optimizer.zero_grad()
        if optimizer_bool is not None:
            optimizer_bool.zero_grad()
        
        output = model(data)
        loss = criterion(output.squeeze(1), target)
        loss.backward()
        
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
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data=torch.gt(data,0.5).float()
            output = model(data)
            correct += (output.squeeze(1) == target).sum().item()
    print(f'\nTest accuracy {correct / len(test_loader.dataset):.4f}')


def main():
    args = get_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    torch.manual_seed(args.seed)
    assert args.lr is None, "lr has no effect in this verion. Remove the setter and use thresh instead"

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
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    
    dataset1 = filter_dataset_by_labels(dataset1)
    dataset2 = filter_dataset_by_labels(dataset2)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    
    fp_params = [x for name,x in model.named_parameters() if 'bool_' not in name]
    optimizer = optim.Adam([x for name,x in model.named_parameters() if 'bool_' not in name], lr=args.lr) if len(fp_params) > 0 else None
    # optimizer_bool = BooleanOptimizer([x for name,x in model.named_parameters() if 'bool_' in name], lr=args.lr)
    optimizer_bool_vanilla = BoldVanillaOptimizer([x for name,x in model.named_parameters() if 'bool_' in name], lr=args.lr, thresh=args.thresh)

    for epoch in range(1, args.epochs + 1):
        # train(args, model, device, train_loader, optimizer, optimizer_bool, epoch)
        train(args, model, device, train_loader, optimizer, optimizer_bool_vanilla, epoch)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_bnn.pt")

if __name__ == '__main__':
    main()