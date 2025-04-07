import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from typing import Any , List , Optional , Callable
from utils import get_args, filter_dataset_by_labels, get_output_dim

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
from bold_loss import XORMismatchLoss, IntScalingLoss, MixtypeXORLoss

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
            self.bool_layers.append(XNORLinear(with_input_output[i], with_input_output[i+1], bool_bprop=False))
            self.actv_layers.append(BoolActvWithThreshDiscrete(0, spread=args.spread))
            # self.actv_layers.append(nn.ReLU())

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        
        # Pass through all layers except the last one
        for i in range(len(self.bool_layers) - 1):
            x = self.bool_layers[i](x)
            x = self.actv_layers[i](x)
        
        # Last layer (no activation after it)
        x = self.bool_layers[-1](x)
        
        return x, None

def parse_arch(arch):
    """Parse architecture string into a list of layer specifications.

    Format: 'conv-CxKxK-S-P,conv-CxKxK-S-P,linear-N'
    
    Where:
    - conv/linear specifies the layer type
    - C is number of output channels (for conv)
    - K is kernel size (for conv)
    - S is stride (for conv)
    - P is padding (for conv) 
    - N is output size (for linear)
    
    Example: 'conv-36x14x14-1-0,conv-36x14x14-1-0,linear-100,linear-10'
    """
    if not arch:
        return []
        
    layers = []
    for layer_spec in arch.split(','):
        parts = layer_spec.split('-')
        layer_type = parts[0]
        
        if layer_type == 'conv':
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
            
        elif layer_type == 'linear':
            # Parse linear layer: N
            out_features = int(parts[1])
            layers.append({
                'type': 'linear',
                'out_features': out_features
            })
            
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

class LogitsConvNet_v2(nn.Module):
    def __init__(self, args):
        super(LogitsConvNet_v2, self).__init__()
        self.bool_layers = nn.ModuleList()
        self.spread = args.spread

        c_in = 1
        H = W = 28
        dim_in = None
        layers = parse_arch(args.arch)
        self.last_conv_layer_index = None
        for i, layer_spec in enumerate(layers):
            if layer_spec['type'] == 'conv':
                layer, output_dim = build_conv_layer(layer_spec, c_in, H, W)
                c_in = layer.out_channels
                H, W = output_dim, output_dim
                self.bool_layers.append(layer)
            elif layer_spec['type'] == 'linear':
                if self.last_conv_layer_index is None:
                    self.last_conv_layer_index = i
                    dim_in = H * W * c_in
                layer, output_dim = build_linear_layer(layer_spec, dim_in)
                dim_in = output_dim
                self.bool_layers.append(layer)

    def forward(self, x):
        for i in range(len(self.bool_layers) - 1):
            if i == self.last_conv_layer_index:
                x = x.view(x.size(0), -1)
            x = self.bool_layers[i](x)
            x = BoolActvWithThreshDiscrete(0, spread=self.spread)(x)
        x = self.bool_layers[-1](x)
        return x, None

class LogitsConvNet(nn.Module):
    def __init__(self, args):
        super(LogitsConvNet, self).__init__()
        C_out = 36
        kH = 14
        kW = 14
        stride = 1
        padding = 0
        dilation = 1
        gd = get_output_dim

        with_input_output = [28*28]+ args.layer_sizes + [len(args.labels)]
        # Create layers dynamically
        self.bool_layers = nn.ModuleList()
        self.actv_layers = nn.ModuleList()
        
        # Create all layers except the last one
        for i in range(len(with_input_output) - 1):
            if i == 0:
                self.bool_layers.append(XNORConv2d(1, C_out, kH, stride=stride, padding=padding, groups=1))
                # self.bool_layers.append(XNORConv2d(1, C_out, kH, padding='same', padding_mode='replicate'))
                self.actv_layers.append(BoolActvWithThreshDiscrete(0, spread=args.spread))
            else:
                dim_out = gd(28, padding, dilation, kH, stride)
                # dim_out = gd(dim_out, kernel_size=2, stride=2)
                self.bool_layers.append(XNORLinear(dim_out ** 2 * C_out, with_input_output[i+1], bool_bprop=False))
                # self.bool_layers.append(XNORLinear(28*28 * C_out, with_input_output[i+1]))
                self.actv_layers.append(BoolActvWithThreshDiscrete(0, spread=args.spread)) 

    def forward(self, x):
        for i in range(len(self.bool_layers) - 1):
            if i == 0:
                x = self.bool_layers[i](x)
                # x = F.max_pool2d(x, 2, stride=2)
                x = x.view(x.size(0), -1)
                x = self.actv_layers[i](x)
            else:
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
    accs = []
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
            accs.append(train_acc / 100.)
            print()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%\tFlips: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), train_acc, batch_flips), end="\t")
            
            # Log statistics using the new log_stats method
            if isinstance(optimizer_bool, BaseBooleanOptimizer) and optimizer_bool is not None:
                optimizer_bool.log_stats()
            
            if args.dry_run:
                break
    print(f'\nAverage accuracy: {sum(accs) / len(accs):.4f}')
    print('\nTotal flips in epoch {}: {}'.format(epoch, total_flips))

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
        print("Use MixtypeXORLoss")
        return MixtypeXORLoss()
    elif args.loss_int_scaling:
        print(f"Use IntScalingLoss with alpha={args.loss_int_scaling_alpha}")
        return IntScalingLoss(alpha=args.loss_int_scaling_alpha)
    else:
        raise ValueError("Choose a loss function from --loss-X")
        
def get_model(args):
    if args.conv_xnor:
        return LogitsConvNet(args)
    elif args.conv_xnor_v2:
        return LogitsConvNet_v2(args)
    elif args.xnor:
        return LogitsNet(args)
    else:
        raise ValueError("Choose an architecture from --conv-xnor or --xnor")

def get_transform(args):
    if args.integer_input:
        steps = args.integer_input_steps
        print(f"Using integer input transformation with {steps} steps")
        return transforms.Compose([
            transforms.ToTensor(),  # This handles the (C,H,W) conversion
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

    # MARK: Transformation of the input

    transform = get_transform(args)
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