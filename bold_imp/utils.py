import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Boolean Deep Learning (BOLD) MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate - not used for boolean parameters, included for compatibility')
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
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    # Dataset arguments
    dataset_group = parser.add_argument_group('Dataset Selection (choose one)')
    dataset_group.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], default='mnist',
                        help='Dataset to use for training and testing (mnist or cifar10)')
    # parser.add_argument('--labels', type=int, nargs='+', default=[1,0],
    #                     help='list of labels to use (default: 1 0)')
    # parser.add_argument('--all-labels', action='store_true', default=False,
    #                     help='Use all labels instead of binary classification')
    # parser.add_argument('--all-labels-cifar', action='store_true', default=False,
    #                     help='Use all labels instead of binary classification for CIFAR')
    # # CIFAR dataset arguments
    # parser.add_argument('--labels-cifar', type=int, nargs='+', default=[0, 1],
    #                     help='list of CIFAR-10 labels to use (default: 0 1)')

    parser.add_argument('--float16', action='store_true', default=False,
                        help='Use float16 precision for computations; Likely buggy; Inspect before using')

    # Model arguments
    parser.add_argument('--spread', type=int, default=None, metavar='N',
                        help='Spread for the activation function')
    parser.add_argument('--spread-std', action='store_true', default=False,
                        help='Use standard deviation of X as base to calculate spread for the activation function')
    parser.add_argument('--spread-fix-ratio', type=float, default=None,
                        help='Use fixed ratio of some parameters of linear layer as spread base')
    
    # parser.add_argument('--layer-sizes', type=int, nargs='+', default=[64],
    #                     help='List of hidden layer sizes (default: [64])')
    parser.add_argument('--activation-range', type=int, nargs='+', default=[-1,1],
                        help='List of output range (default: [-1,1])')

    # Input transformation arguments
    transform_group = parser.add_argument_group('Input Transformation Options')
    transform_group.add_argument('--input-int', action='store_true', default=False,
                        help='Use integer input transformation (centered around zero)')
    transform_group.add_argument('--integer-int-steps', type=int, default=255,
                        help='Number of steps for integer input transformation (default: 255)')
    transform_group.add_argument('--input-grayscale', action='store_true', default=False,
                        help='Use grayscale input transformation')
    transform_group.add_argument('--input-grayscale-steps', type=int, default=255,
                        help='Number of steps for grayscale input transformation (default: 255)')
    transform_group.add_argument('--input-augmentation', action='store_true', default=False,
                        help='Use input augmentation')

    # architecture arguments mutually exclusive
    architecture_group = parser.add_argument_group('Architecture Selection (choose one)')
    architecture_group.add_argument('--arch-custom', action='store_true', default=False,
                        help='Use convolutional network v2')
    architecture_group.add_argument('--arch', type=str, default='',
                        help='Architecture specification for conv-xnor-v2')

    # Optimizer selection arguments - mutually exclusive
    optimizer_group = parser.add_argument_group('Optimizer Selection (choose one)')
    optimizer_group.add_argument('--opt-momentum', action='store_true', default=False,
                        help='Use momentum-based boolean optimizer')
    optimizer_group.add_argument('--opt-probabilistic', action='store_true', default=False,
                        help='Use probabilistic boolean optimizer (linear flip probability from 0 to threshold)')
    
    # Momentum parameters
    momentum_group = parser.add_argument_group('Momentum Parameters (used with --opt-momentum)')
    momentum_group.add_argument('--opt-momentum-val', type=float, default=0.9,
                        help='Momentum factor (default: 0.9)')
    momentum_group.add_argument('--opt-momentum-dampening', type=float, default=0.0,
                        help='Dampening factor for momentum (default: 0.0)')
                        
    # Probabilistic parameters
    prob_group = parser.add_argument_group('Probabilistic Parameters (used with --opt-probabilistic)')
    prob_group.add_argument('--prob-flip-ratio', type=float, default=0.001,
                        help='Target percentage of parameters to flip per step (default: 0.001, i.e., 0.1%%)')
    prob_group.add_argument('--prob-flip-ratio-decay', type=float, default=0.9,
                        help='Decay rate for flip ratio per epoch (default: 0.9)')
    prob_group.add_argument('--prob-flip-ratio-decay-epochs', type=int, default=10,
                        help='Number of epochs to decay flip ratio (default: 10)')
    prob_group.add_argument('--prob-flip-ratio-batch-acc-aware', type=float, default=0.3,
                        help='Scale flip ratio based on batch accuracy (default: 0.3)')

    # Loss function arguments
    loss_group = parser.add_mutually_exclusive_group()
    loss_group.add_argument('--loss-naive', action='store_true', default=False,
                        help='Use naive XOR mismatch loss')
    loss_group.add_argument('--loss-int-scaling', action='store_true', default=False,
                        help='Use integer scaling loss')
    loss_group.add_argument('--loss-cross-entropy', action='store_true', default=False,
                        help='Use cross entropy loss')
    parser.add_argument('--loss-int-scaling-alpha', type=int, default=1000, metavar='N',
                        help='Alpha for integer scaling loss')
    loss_group.add_argument('--loss-int-l1', action='store_true', default=False,
                            help='Use int L1 loss. This one requires the last layer to be activation with range.')
    return parser.parse_args()

def filter_dataset_by_labels(dataset, wanted_labels, debug=False):
    # Create a mapping from original labels to new sequential indices
    label_to_idx = {label: idx for idx, label in enumerate(wanted_labels)}
    
    # Filter dataset to only include wanted labels
    mask = torch.tensor([label in wanted_labels for label in dataset.targets])
    if not isinstance(dataset.targets, torch.Tensor):
        dataset.targets = torch.tensor(dataset.targets)
    dataset.data = dataset.data[mask]
    dataset.targets = dataset.targets[mask]

    if debug:
        print("Pre-reindexing targets: ", dataset.targets[:10].tolist())
        # Visualize the first 10 samples with their labels as captions
        import matplotlib.pyplot as plt
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        axes = axes.flatten()
        
        # Plot the first 10 samples
        for i in range(min(10, len(dataset.data))):
            # Get the image and label
            img = dataset.data[i].numpy()
            label = dataset.targets[i].item() if isinstance(dataset.targets[i], torch.Tensor) else dataset.targets[i]
            
            # Display the image
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Label: {label} should map to {label_to_idx[label]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Map the original labels to sequential indices [0,1,2,...]
    # Convert tensor values to Python integers for dictionary lookup
    dataset.targets = torch.tensor([label_to_idx[label.item()] for label in dataset.targets])
    
    if debug:
        # print the first 10 targets
        print("Post-reindexing targets: ", dataset.targets[:10].tolist())
    
    return dataset

def get_output_dim(length, padding = 0, dilation = 1, kernel_size = 1, stride = 1):
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

def get_tensor_stats(tensor, counter=0):
    """Compute basic statistics of a tensor.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Dictionary containing min, max, mean, median and count statistics
    """
    # Initialize counter if it doesn't exist
    if not hasattr(get_tensor_stats, 'counter'):
        get_tensor_stats.counter = counter  # Start with 5 calls
    
    # Only compute stats if counter > 0
    if get_tensor_stats.counter > 0:
        stats = {
            'min': tensor.min().item(),
            'max': tensor.max().item(), 
            'mean': tensor.mean().item(),
            'median': tensor.median().item(),
            'count': tensor.numel()
        }
        get_tensor_stats.counter -= 1  # Decrement counter
        print(stats)

def print_important_args(args):
    # Meta information
    print("Meta information:")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Seed: {args.seed}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Test batch size: {args.test_batch_size}")
    print()
    
    # input transformation
    print("Input configuration:")
    if args.input_int:
        print(f"  - Integer input transformation")
        print(f"    - Steps: {args.integer_int_steps}")
    elif args.input_grayscale:
        print(f"  - Grayscale input transformation")
        print(f"    - Steps: {args.input_grayscale_steps}")
    else:
        print(f"  - No input transformation")

    print(f"  - Input augmentation: {args.input_augmentation}")
    print()

    # architecture
    print("Architecture:")
    print()
    if args.arch_custom:
        for layer in args.arch.split(','):
            print(f"{layer}")
    print()
    # optimizer
    print("Optimizer:")
    print(f"  - Momentum: {args.opt_momentum}")
    print(f"    - value: {args.opt_momentum_val}")
    print(f"    - Dampening: {args.opt_momentum_dampening}")
    print(f"  - Probabilistic optimizer: {args.opt_probabilistic}")
    print(f"    - Flip ratio: {args.prob_flip_ratio}")
    print(f"    - Flip ratio decay: {args.prob_flip_ratio_decay}")
    print(f"    - Flip ratio decay epochs: {args.prob_flip_ratio_decay_epochs}")
    print(f"    - Flip ratio batch acc aware: {args.prob_flip_ratio_batch_acc_aware}")
    print()
    # loss function
    print("Loss function:")
    if args.loss_int_l1:
        print(f"  - Int L1 loss")
    elif args.loss_int_scaling:
        print(f"  - Integer scaling loss")
    elif args.loss_cross_entropy:
        print(f"  - Cross entropy loss")
    print()

    # activation function
    print("Bool Activation function config:")
    if args.spread:
        print(f"  - fixed backward spread: {args.spread}")
    if args.spread_std:
        print(f"  - backward spread via input std: {args.spread_std}")
    if args.spread_fix_ratio:
        print(f"  - backward spread fix-ratio: {args.spread_fix_ratio}")
    if args.activation_range:
        print(f"  - Default activation range: {args.activation_range}")
    print()


if __name__ == "__main__":
    args = get_args()
 

