import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
from torchvision import datasets, transforms
from typing import Any , List , Optional , Callable
from utils import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib
import platform
import os



# Set print options to show full tensor contents
torch.set_printoptions(profile="full")

from bold_layers import BoolActvWithThreshDiscrete, XNORLinear, XNORConv2d, ANDLinear, XORLinear, XNORLinearManual
from bold_opt import (
    BaseBooleanOptimizer,
    create_vanilla_optimizer,
    create_momentum_optimizer,
    create_probabilistic_optimizer,
    create_probabilistic_momentum_optimizer
)
from bold_loss import XORMismatchLoss, IntScalingLoss, MixtypeXORLoss, IntL1Loss


def parse_arch(arch):
    """Parse architecture string into a list of layer specifications.

    Format: 'conv-CxKxK-S-P,conv-CxKxK-S-P,linear-N,activation-type'
    
    Where:
    - conv/linear/activation specifies the layer type
    - C is number of output channels (for conv)
    - K is kernel size (for conv)
    - S is stride (for conv)
    - P is padding (for conv) 
    - N is output size (for linear)
    - type is the activation function type (for activation)
    
    Example: 'conv-36x14x14-1-0,conv-36x14x14-1-0,activation-relu,linear-100,linear-10'
    """
    if not arch:
        return []
        
    layers = []
    for layer_spec in arch.split(','):
        parts = layer_spec.split('-')
        layer_type = parts[0]
        
        if layer_type == 'conv' or layer_type == 'c':
            # Parse conv layer: CxKxK-S-P
            channels, *kernel = parts[1].split('x')
            stride = int(parts[2])
            padding = int(parts[3])
            
            layers.append({
                'type': 'conv',
                'out_channels': int(channels),
                'kernel_size': (int(kernel[0]), int(kernel[1])),
                'stride': stride,
                'padding': padding
            })
            
        elif layer_type == 'linear' or layer_type == 'l':
            # Parse linear layer: N
            out_features = int(parts[1])
            layers.append({
                'type': 'linear',
                'out_features': out_features
            })
            
        elif layer_type == 'activation' or layer_type == 'a':
            # Parse activation layer: type
            activation_type = parts[1]
            if activation_type == 'int' and len(parts) > 3:
                # Handle activation-int-(low)-(high) format
                # Extract the range values, handling negative numbers
                low_str = parts[2].strip('()')
                high_str = parts[3].strip('()')
                
                # Convert 'neg' to '-' for negative numbers
                low_str = low_str.replace('neg', '-')
                high_str = high_str.replace('neg', '-')
                
                low = float(low_str)
                high = float(high_str)
                
                layers.append({
                    'type': 'activation',
                    'activation_type': activation_type,
                    'range': (low, high)
                })
            else:
                # Handle normal activation-int format
                layers.append({
                    'type': 'activation',
                    'activation_type': activation_type
                })
        elif layer_type == 'pool' or layer_type == 'p':
            # Parse pooling layer: type-KxK-S-P
            # Example: pool-max-2x2-2-0 or pool-avg-3x3-1-1
            pool_type = parts[1]
            kernel = [int(x) for x in parts[2].split('x')]
            stride = int(parts[3])
            padding = int(parts[4])
            layers.append({
                'type': 'pool',
                'pool_type': pool_type,
                'kernel_size': (kernel[0], kernel[1]),
                'stride': stride,
                'padding': padding
            })
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
    return layers

def build_conv_layer(layer_spec, c_in, H, W):
    assert layer_spec['type'] == 'conv'
    # unpack
    c_out = layer_spec['out_channels']
    kH = layer_spec['kernel_size'][0]
    kW = layer_spec['kernel_size'][1]
    stride = layer_spec['stride']
    padding = layer_spec['padding']
    return XNORConv2d(c_in, c_out, (kH, kW), stride=stride, padding=padding, groups=1), get_output_dim(H, padding, 1, kH, stride)

def build_linear_layer(layer_spec, dim_in):
    assert layer_spec['type'] == 'linear'
    dim_out = layer_spec['out_features']
    return XNORLinear(dim_in, dim_out, bool_bprop=False), dim_out

def build_activation_layer(layer_spec, args):
    """Build an activation layer based on the specification.
    
    Args:
        layer_spec: Dictionary containing layer specifications
        args: Command line arguments
        
    Returns:
        Activation layer module
    """
    assert layer_spec['type'] == 'activation'
    activation_type = layer_spec['activation_type'].lower()
    
    if activation_type == 'relu':
        return nn.ReLU()
    elif activation_type == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation_type == 'sigmoid':
        return nn.Sigmoid()
    elif activation_type == 'tanh':
        return nn.Tanh()
    elif activation_type == 'int':
        # Check if custom range is provided
        if 'range' in layer_spec:
            low, high = layer_spec['range']
            return BoolActvWithThreshDiscrete(0, spread=args.spread, output_range=(low, high))
        else:
            # Use default range from args
            return BoolActvWithThreshDiscrete(0, spread=args.spread, output_range=args.activation_range)
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")
    
def build_pool_layer(layer_spec, c_in, H, W):
    """Build a pooling layer based on the specification.
    
    Args:
        layer_spec: Dictionary containing layer specifications
        c_in: Number of input channels
        H: Input height
        W: Input width
        
    Returns:
        Tuple of (pooling layer, output dimension)
    """
    assert layer_spec['type'] == 'pool'
    pool_type = layer_spec['pool_type'].lower()
    kernel_size = layer_spec['kernel_size']
    stride = layer_spec['stride'] 
    padding = layer_spec['padding']

    if pool_type == 'max':
        layer = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
    elif pool_type == 'avg':
        layer = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
    else:
        raise ValueError(f"Unsupported pooling type: {pool_type}")

    output_dim = get_output_dim(H, padding, 1, kernel_size[0], stride)
    return layer, output_dim

class CustomIntNet(nn.Module):
    def __init__(self, args):
        super(CustomIntNet, self).__init__()
        self.args = args
        self.bool_layers = nn.ModuleList()
        if args.dataset == 'cifar10':
            c_in = 1 if args.input_grayscale else 3
            H = W = 32
        else:
            c_in = 1
            H = W = 28

        dim_in = None
        layers = parse_arch(args.arch)
        self.first_linear_layer_index = None
        for i, layer_spec in enumerate(layers):
            if layer_spec['type'] == 'conv':
                layer, output_dim = build_conv_layer(layer_spec, c_in, H, W)
                c_in = layer.out_channels
                H, W = output_dim, output_dim
                self.bool_layers.append(layer)
            elif layer_spec['type'] == 'linear':
                if self.first_linear_layer_index is None:
                    self.first_linear_layer_index = i
                    dim_in = H * W * c_in
                layer, output_dim = build_linear_layer(layer_spec, dim_in)
                dim_in = output_dim
                self.bool_layers.append(layer)
            elif layer_spec['type'] == 'activation':
                activation_layer = build_activation_layer(layer_spec, args)
                self.bool_layers.append(activation_layer)
            elif layer_spec['type'] == 'pool':
                layer, output_dim = build_pool_layer(layer_spec, c_in, H, W)
                c_in = c_in # pooling layer does not change the number of channels
                H, W = output_dim, output_dim
                self.bool_layers.append(layer)

            else:
                raise ValueError(f"Unsupported layer type: {layer_spec['type']}")

    def forward(self, x):
        for i in range(len(self.bool_layers)):
            if i == self.first_linear_layer_index:
                x = x.view(x.size(0), -1)
            x = self.bool_layers[i](x)
            if self.args.loss_cross_entropy:
                x = F.log_softmax(x, dim=1)
        return x, None


def is_loss_gradient_boolean(args):
    assert args.activate_before_output == False, "Has no effect when model applies activation before output. Please double check."
    if args.loss_int_scaling:
        return False
    elif args.loss_naive:
        return False
    elif args.loss_cross_entropy:
        return False
    else:
        raise ValueError("Choose a loss function from --loss-X")

class AccuracyVisualizer:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.accuracies = []
        self.batch_indices = []
        self.current_epoch = 1
        
        # Set up the plot
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.line, = self.ax.plot([], [], 'b-', label='Batch Accuracy')
        
        # Configure the plot
        self.ax.set_xlabel('Batch')
        self.ax.set_ylabel('Accuracy')
        self.ax.set_title(f'Training Accuracy (Epoch {self.current_epoch})')
        self.ax.set_ylim(0, 1.0)
        self.ax.grid(True)
        self.ax.legend()
        
        plt.tight_layout()
        plt.show(block=False)
    
    def update(self, batch_idx, accuracy, epoch=None):
        if epoch is not None and epoch != self.current_epoch:
            # Reset for new epoch
            self.current_epoch = epoch
            self.accuracies = []
            self.batch_indices = []
            self.ax.set_title(f'Training Accuracy (Epoch {self.current_epoch})')
        
        # Add new data point
        self.accuracies.append(accuracy)
        self.batch_indices.append(batch_idx)
        
        # Limit number of displayed points
        if len(self.accuracies) > self.max_points:
            self.accuracies = self.accuracies[-self.max_points:]
            self.batch_indices = self.batch_indices[-self.max_points:]
        
        # Update data in the plot
        self.line.set_data(self.batch_indices, self.accuracies)
        
        # Adjust x-axis limits to show all data
        if self.batch_indices:
            self.ax.set_xlim(min(self.batch_indices), max(self.batch_indices) + 1)
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def close(self):
        plt.close(self.fig)

# Modified train function with real-time visualization
def train(args, model, device, train_loader, optimizer, optimizer_bool, epoch):
    model.train()
    total_flips = 0
    criterion = get_criterion(args)
    accs = []
    
    # Calculate the width needed for the largest value (total dataset size)
    progress_width = len(str(len(train_loader.dataset)))
    
    # Initialize visualizer on first epoch
    # if epoch == 1:
    #     if not hasattr(train, 'visualizer'):
    #         train.visualizer = AccuracyVisualizer()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        zero_grads_for([optimizer, optimizer_bool])
        output, _ = model(data)
        loss = criterion(output, target)
        loss.backward()

        pred = torch.argmax(output, dim=1)
        num_correct = pred.eq(target).sum().item()
        num_total = len(target)
        config.hooks['cur_acc'] = (num_correct, num_total)
        
        # Calculate accuracy for every batch
        batch_acc = num_correct / num_total
        
        # Update the visualization with every batch
        # train.visualizer.update(batch_idx, batch_acc, epoch)
        
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
            train_acc = 100. * pred.eq(target).sum().item() / len(target)
            accs.append(train_acc / 100.)
            print()
            print('Epoch: {} [{:>{width}}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%\tFlips: {}'.format(
                epoch, 
                batch_idx * len(data), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss.item(), 
                train_acc, 
                batch_flips,
                width=progress_width), 
                end="")
            if isinstance(optimizer_bool, BaseBooleanOptimizer) and optimizer_bool is not None:
                optimizer_bool.log_stats()
            
            if args.dry_run:
                break
            
    print(f'\nAv. Acc: {sum(accs) / len(accs):.4f}')
    print('\nFlips in epoch {}: {}'.format(epoch, total_flips))

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
    if args.loss_naive:
        return MixtypeXORLoss()
    elif args.loss_int_scaling:
        return IntScalingLoss(alpha=args.loss_int_scaling_alpha)
    elif args.loss_cross_entropy:
        return F.nll_loss
    elif args.loss_int_l1:
        return IntL1Loss(activation_range=(args.loss_int_l1_scale[0], args.loss_int_l1_scale[1]))
    else:
        raise ValueError("Choose a loss function from --loss-X")
        
def get_model(args):
    if args.arch_custom:
        return CustomIntNet(args)
    else:
        raise ValueError("Choose an architecture from --arch-custom")

def get_transform(args):
    if config.args.float16:
        compress = lambda x: x.to(torch.float16)
    else:
        compress = lambda x: x

    if args.input_int and args.dataset == 'cifar10' and args.input_augmentation:
        steps = args.integer_int_steps
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness = 0.1,contrast = 0.1,saturation = 0.1),
            transforms.RandomAdjustSharpness(sharpness_factor = 2,p = 0.2),
            transforms.ToTensor() ,
            transforms.Lambda(lambda x: x * steps),  # Convert to [0,255]
            transforms.Lambda(lambda x: x - steps / 2),  # Center around zero: [-127.5, 127.5]
            transforms.Lambda(lambda x: torch.floor(x)),  # Floor to get integer values: [-127, 127]
            transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=0.0, inplace=False),
        ])
    elif args.input_int:
        steps = args.integer_int_steps
        return transforms.Compose([
            transforms.ToTensor(),  # This handles the (C,H,W) conversion
            transforms.Lambda(lambda x: x * steps),  # Convert to [0,255]
            transforms.Lambda(lambda x: x - steps / 2),  # Center around zero: [-127.5, 127.5]
            transforms.Lambda(lambda x: torch.floor(x)),  # Floor to get integer values: [-127, 127]
            transforms.Lambda(compress)
        ])
    elif args.input_grayscale:
        steps = args.input_grayscale_steps
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * steps),  # Convert to [0,255]
            transforms.Lambda(lambda x: x - steps / 2),  # Center around zero: [-127.5, 127.5]
            transforms.Lambda(lambda x: torch.floor(x))  # Floor to get integer values: [-127, 127]
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2. * torch.gt(x, 0.5).float() - 1.)  # Add thresholding to transformation pipeline
        ])

def main():
    args = get_args()
    print_important_args(args)
    config.args = args


    # Try to set an appropriate backend based on platform
    try:
        if platform.system() == 'Darwin':  # macOS
            matplotlib.use('macosx')
        elif platform.system() == 'Linux':
            # Try TkAgg first on Linux if in a desktop environment
            # Fall back to Agg for headless servers
            if 'DISPLAY' in os.environ:
                matplotlib.use('TkAgg')
            else:
                matplotlib.use('Agg')
        else:
            matplotlib.use('TkAgg')  # Try TkAgg on other systems
    except ImportError:
        # If the preferred backend isn't available, try a more universal one
        try:
            matplotlib.use('WebAgg')  # Browser-based display
        except ImportError:
            matplotlib.use('Agg')  # Fallback to non-interactive
            print("Warning: Using non-interactive Agg backend. Plots will be saved to files only.")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    torch.manual_seed(args.seed)
    # assert args.lr is None, "lr has no effect in this verion. Remove the setter and use thresh instead"

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

    transform = get_transform(args)

    if args.dataset == 'cifar10':
        dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transform)
        dataset2 = datasets.CIFAR10('../data', train=False,
                       transform=transform)
        # dataset1 = filter_dataset_by_labels(dataset1, wanted_labels=args.labels)
        # dataset2 = filter_dataset_by_labels(dataset2, wanted_labels=args.labels)
    elif args.dataset == 'mnist':
        dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
        dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)
        # dataset1 = filter_dataset_by_labels(dataset1, wanted_labels=args.labels)
        # dataset2 = filter_dataset_by_labels(dataset2, wanted_labels=args.labels)
    else:
        raise ValueError("Choose a dataset from --dataset mnist or --dataset cifar10")

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = get_model(args).to(device)
    
    fp_params = [x for name,x in model.named_parameters() if 'bool_' not in name]
    # fp_params = [x for name,x in model.named_parameters() ]
    optimizer = optim.Adam(fp_params, lr=args.lr) if len(fp_params) > 0 else None

    bool_params = [x for name,x in model.named_parameters() if 'bool_' in name]
    
    # Create the optimizer with the selected configuration
    if args.opt_probabilistic:
        if args.opt_momentum:
            optimizer_bool = create_probabilistic_momentum_optimizer(
                bool_params,
                lr=args.lr,
                momentum=args.opt_momentum_val,
                dampening=args.opt_momentum_dampening,
                flip_ratio=args.prob_flip_ratio
            )
            # optimizer_bool = None
        else:
            optimizer_bool = create_probabilistic_optimizer(
                bool_params,
                lr=args.lr,
                flip_ratio=args.prob_flip_ratio
            )
    elif args.opt_momentum:
        optimizer_bool = create_momentum_optimizer(
            bool_params,
            lr=args.lr,
            thresh=args.thresh,
            momentum=args.opt_momentum_val,
            dampening=args.opt_momentum_dampening
        )
    else:
        optimizer_bool = create_vanilla_optimizer(
            bool_params,
            lr=args.lr,
            thresh=args.thresh
        )

    for epoch in range(1, args.epochs + 1):
        config.hooks['epoch'] = epoch
        train(args, model, device, train_loader, optimizer, optimizer_bool, epoch)
        test(args, model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_bnn.pt")

if __name__ == '__main__':
    main()