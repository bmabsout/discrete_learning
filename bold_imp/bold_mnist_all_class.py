import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from typing import Any , List , Optional , Callable
from utils import get_args, filter_dataset_by_labels

from bold_layers import XORLinear, BoolActvWithThreshDiscrete
from bold_opt import BoldProbabilisticOptimizer, BoldVanillaMomentumOptimizer, BoldVanillaOptimizer, BooleanOptimizer
from bold_loss import XORMismatchLoss, MixtypeXORLoss, IntScalingLoss

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.bool_fc1 = XORLinear(28*28, 64,bool_bprop=False)
        self.actv1 = BoolActvWithThreshDiscrete(28*28, spread=args.spread)
        self.bool_fc2 = XORLinear(64, len(args.labels),bool_bprop=True)  # well, bool_bprop should be False. They both train regardless
        self.actv2 = BoolActvWithThreshDiscrete(64, spread=args.spread)

    def forward(self, x):
        x = x.reshape(-1,28*28)
        x = self.bool_fc1(x)
        x = self.actv1(x)
        x = self.bool_fc2(x)
        fork = x.detach()
        x = self.actv2(x)
        return x, fork

class LogitsNet(nn.Module):
    def __init__(self, args):
        super(LogitsNet, self).__init__()
        self.bool_fc1 = XORLinear(28*28, 64,bool_bprop=False)
        self.actv1 = BoolActvWithThreshDiscrete(28*28, spread=args.spread)
        self.bool_fc2 = XORLinear(64, len(args.labels),bool_bprop=is_loss_gradient_boolean(args))  

    def forward(self, x):
        x = x.reshape(-1,28*28)
        x = self.bool_fc1(x)
        x = self.actv1(x)
        x = self.bool_fc2(x)
        return x, None

def is_loss_gradient_boolean(args):
    assert args.activate_before_output == False, "Has no effect when model applies activation before output. Please double check."
    if args.loss_int_scaling:
        return False
    elif args.loss_naive:
        return True
    else:
        raise ValueError("Choose a loss function from --loss-X")

def train(args, model, device, train_loader, optimizer, optimizer_bool, epoch):
    model.train()
    total_flips = 0
    criterion = get_criterion(args)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        zero_grads_for([optimizer, optimizer_bool])
        output, _ = model(data)
        loss = criterion(output, target)
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

def test(args, model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, fork = model(data)
            pred = torch.argmax(output, dim=1)
            correct += (pred == target).sum().item()
    print(f'Test accuracy {correct / len(test_loader.dataset):.4f}\n')

def zero_grads_for(opts):
    for opt in opts:
        if opt is not None:
            opt.zero_grad()

def step_grads_for(opts):
    for opt in opts:
        if opt is not None:
            opt.step()

def get_criterion(args):
    if args.activate_before_output:
        print("Model applies activation before output. Overwrite criterion to XORMismatchLoss")
        return XORMismatchLoss()
    else:
        if args.loss_naive:
            print("Use MixtypeXORLoss")
            return MixtypeXORLoss()
        elif args.loss_int_scaling:
            print(f"Use IntScalingLoss with alpha={args.loss_int_scaling_alpha}")
            return IntScalingLoss(alpha=args.loss_int_scaling_alpha)
        else:
            raise ValueError("Choose a loss function from --loss-X")
        
def get_model(args):
    if args.activate_before_output:
        return Net(args)
    else:
        return LogitsNet(args)

def main():
    args = get_args()
    if args.all_labels:
        args.labels = range(10)
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
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.gt(x, 0.5).float())  # Add thresholding to transformation pipeline
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    dataset1 = filter_dataset_by_labels(dataset1, wanted_labels=args.labels)
    dataset2 = filter_dataset_by_labels(dataset2, wanted_labels=args.labels)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = get_model(args).to(device)
    
    fp_params = [x for name,x in model.named_parameters() if 'bool_' not in name]
    optimizer = optim.Adam(fp_params, lr=args.lr) if len(fp_params) > 0 else None

    if args.use_momentum:
        optimizer_bool = BoldVanillaMomentumOptimizer([x for name,x in model.named_parameters() if 'bool_' in name], lr=args.lr, thresh=args.thresh, momentum=args.momentum, dampening=args.dampening)
    elif args.use_probabilistic:
        optimizer_bool = BoldProbabilisticOptimizer([x for name,x in model.named_parameters() if 'bool_' in name], lr=args.lr, thresh=args.thresh)
    else:
        optimizer_bool = BoldVanillaOptimizer([x for name,x in model.named_parameters() if 'bool_' in name], lr=args.lr, thresh=args.thresh)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, optimizer_bool, epoch)
        test(args, model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_bnn.pt")

if __name__ == '__main__':
    main()