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
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
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
    
    # Boolean optimizer arguments
    parser.add_argument('--thresh', type=int, default=150, metavar='N',
                        help='Threshold for weight flipping. Higher values make flipping less likely. '
                             'For vanilla optimizer: flips when grad > thresh. '
                             'For probabilistic: 100%% flip probability at threshold.')
    
    # Dataset arguments
    parser.add_argument('--labels', type=int, nargs='+', default=[1,0],
                        help='list of labels to use (default: 1 0)')
    parser.add_argument('--all-labels', action='store_true', default=False,
                        help='Use all labels instead of binary classification')
    parser.add_argument('--all-labels-cifar', action='store_true', default=False,
                        help='Use all labels instead of binary classification for CIFAR')
    
    # Model arguments
    parser.add_argument('--spread', type=int, default=10, metavar='N',
                        help='Spread for the activation function')
    parser.add_argument('--use-relu', action='store_true', default=False,
                        help='Use ReLU activation instead of boolean activation')
    # parser.add_argument('--activate-before-output', action='store_true', default=False,
    #                     help='Apply activation function before the output layer')
    parser.add_argument('--layer-sizes', type=int, nargs='+', default=[64],
                        help='List of hidden layer sizes (default: [64])')

    # Input transformation arguments
    transform_group = parser.add_argument_group('Input Transformation Options')
    transform_group.add_argument('--integer-input', action='store_true', default=False,
                        help='Use integer input transformation (centered around zero)')
    transform_group.add_argument('--integer-input-steps', type=int, default=255,
                        help='Number of steps for integer input transformation (default: 255)')
    transform_group.add_argument('--input-grayscale', action='store_true', default=False,
                        help='Use grayscale input transformation')
    transform_group.add_argument('--input-grayscale-steps', type=int, default=255,
                        help='Number of steps for grayscale input transformation (default: 255)')

    # architecture arguments mutually exclusive
    architecture_group = parser.add_argument_group('Architecture Selection (choose one)')
    architecture_group.add_argument('--conv-xnor', action='store_true', default=False,
                        help='Use convolutional network')
    architecture_group.add_argument('--conv-xnor-v2', action='store_true', default=False,
                        help='Use convolutional network v2')
    architecture_group.add_argument('--xnor', action='store_true', default=False,
                        help='Use XNOR network')
    architecture_group.add_argument('--arch', type=str, default='',
                        help='Architecture specification for conv-xnor-v2')

    # Optimizer selection arguments - mutually exclusive
    optimizer_group = parser.add_argument_group('Optimizer Selection (choose one)')
    optimizer_group.add_argument('--use-momentum', action='store_true', default=False,
                        help='Use momentum-based boolean optimizer')
    optimizer_group.add_argument('--use-probabilistic', action='store_true', default=False,
                        help='Use probabilistic boolean optimizer (linear flip probability from 0 to threshold)')
    
    # Momentum parameters
    momentum_group = parser.add_argument_group('Momentum Parameters (used with --use-momentum)')
    momentum_group.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum factor (default: 0.9)')
    momentum_group.add_argument('--dampening', type=float, default=0.0,
                        help='Dampening factor for momentum (default: 0.0)')
                        
    # Probabilistic parameters
    prob_group = parser.add_argument_group('Probabilistic Parameters (used with --use-probabilistic)')
    prob_group.add_argument('--flip-ratio', type=float, default=0.001,
                        help='Target percentage of parameters to flip per step (default: 0.001, i.e., 0.1%%)')

    # Loss function arguments
    loss_group = parser.add_mutually_exclusive_group()
    loss_group.add_argument('--loss-naive', action='store_true', default=False,
                        help='Use naive XOR mismatch loss')
    loss_group.add_argument('--loss-int-scaling', action='store_true', default=False,
                        help='Use integer scaling loss')
    parser.add_argument('--loss-int-scaling-alpha', type=int, default=1000, metavar='N',
                        help='Alpha for integer scaling loss')
    return parser.parse_args()

def filter_dataset_by_labels(dataset, wanted_labels, debug=False):
    # Create a mapping from original labels to new sequential indices
    label_to_idx = {label: idx for idx, label in enumerate(wanted_labels)}
    
    # Filter dataset to only include wanted labels
    mask = torch.tensor([label in wanted_labels for label in dataset.targets])
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



if __name__ == "__main__":
    args = get_args()
    # print(args.labels)
 

