import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from typing import Any , List , Optional , Callable
from utils import get_args, filter_dataset_by_labels

from bold_layers import MixtypeXORLinear, XORLinear, BoolActvWithThreshDiscrete
from bold_opt import (
    BaseBooleanOptimizer,
    create_vanilla_optimizer,
    create_momentum_optimizer,
    create_probabilistic_optimizer,
    create_probabilistic_momentum_optimizer
)
from bold_loss import XORMismatchLoss, MixtypeXORLoss, IntScalingLoss

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        # Define layer sizes with input and output dimensions
        with_input_output = [28*28]+ args.layer_sizes + [len(args.labels)]
        
        # Create layers dynamically
        self.bool_layers = nn.ModuleList()
        self.actv_layers = nn.ModuleList()
        
        # Create all layers
        for i in range(len(with_input_output) - 1):
            # Last layer uses bool_bprop=True, others use False
            bool_bprop = (i == len(with_input_output) - 2)
            self.bool_layers.append(XORLinear(with_input_output[i], with_input_output[i+1], bool_bprop=bool_bprop))
            self.actv_layers.append(BoolActvWithThreshDiscrete(with_input_output[i], spread=args.spread))

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        
        # Pass through all layers except the last one
        for i in range(len(self.bool_layers)):
            x = self.bool_layers[i](x)
            # Before the last activation, create a fork for the output
            if i == len(self.bool_layers) - 1:
                fork = x.detach()
            x = self.actv_layers[i](x)
        
        return x, fork

class LogitsNet(nn.Module):
    def __init__(self, args):
        super(LogitsNet, self).__init__()
        # Define layer sizes with input and output dimensions

        with_input_output = [28*28]+ args.layer_sizes + [len(args.labels)]
        # Create layers dynamically
        self.bool_layers = nn.ModuleList()
        self.actv_layers = nn.ModuleList()
        
        # Create all layers except the last one
        for i in range(len(with_input_output) - 1):
            self.bool_layers.append(XORLinear(with_input_output[i], with_input_output[i+1], bool_bprop=False))
            # For activation, we use the input size for the spread parameter
            self.actv_layers.append(BoolActvWithThreshDiscrete(with_input_output[i], spread=args.spread))

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        
        # Pass through all layers except the last one
        for i in range(len(self.bool_layers) - 1):
            x = self.bool_layers[i](x)
            x = self.actv_layers[i](x)
        
        # Last layer (no activation after it)
        x = self.bool_layers[-1](x)
        
        return x, None

def is_loss_gradient_boolean(args):
    assert args.activate_before_output == False, "Has no effect when model applies activation before output. Please double check."
    if args.loss_int_scaling:
        return False
    elif args.loss_naive:
        return False
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
            pred = torch.argmax(output, dim=1)
            train_acc = 100. * pred.eq(target).sum().item() / len(target)
            print()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%\tFlips: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), train_acc, batch_flips), end="\t")
            
            # Log statistics using the new log_stats method
            if isinstance(optimizer_bool, BaseBooleanOptimizer) and optimizer_bool is not None:
                optimizer_bool.log_stats()
            
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

    bool_params = [x for name,x in model.named_parameters() if 'bool_' in name]
    
    # Create the optimizer with the selected configuration
    if args.use_probabilistic:
        if args.use_momentum:
            print(f"Using probabilistic momentum optimizer with flip ratio={args.flip_ratio:.6f}, "
                  f"momentum={args.momentum}, dampening={args.dampening}")
            optimizer_bool = create_probabilistic_momentum_optimizer(
                bool_params,
                lr=args.lr,
                momentum=args.momentum,
                dampening=args.dampening,
                flip_ratio=args.flip_ratio
            )
        else:
            print(f"Using probabilistic optimizer with flip ratio={args.flip_ratio:.6f}")
            optimizer_bool = create_probabilistic_optimizer(
                bool_params,
                lr=args.lr,
                flip_ratio=args.flip_ratio
            )
    elif args.use_momentum:
        print(f"Using momentum optimizer with threshold={args.thresh}, "
              f"momentum={args.momentum}, dampening={args.dampening}")
        optimizer_bool = create_momentum_optimizer(
            bool_params,
            lr=args.lr,
            thresh=args.thresh,
            momentum=args.momentum,
            dampening=args.dampening
        )
    else:
        print(f"Using vanilla optimizer with threshold={args.thresh}")
        optimizer_bool = create_vanilla_optimizer(
            bool_params,
            lr=args.lr,
            thresh=args.thresh
        )

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, optimizer_bool, epoch)
        test(args, model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_bnn.pt")

if __name__ == '__main__':
    main()