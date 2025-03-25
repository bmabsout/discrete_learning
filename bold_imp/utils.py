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
                             'For probabilistic: 100% flip probability at threshold.')
    
    # Dataset arguments
    parser.add_argument('--labels', type=int, nargs='+', default=[1,0],
                        help='list of labels to use (default: 1 0)')
    parser.add_argument('--all-labels', action='store_true', default=False,
                        help='Use all labels instead of binary classification')
    
    # Model arguments
    parser.add_argument('--spread', type=int, default=10, metavar='N',
                        help='Spread for the activation function')
    parser.add_argument('--logits-output', action='store_true', default=False,
                        help='Use logits output instead of boolean output')
    
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
    
    return parser.parse_args()

def filter_dataset_by_labels(dataset, wanted_labels):
    # Create a mapping from original labels to new sequential indices
    label_to_idx = {label: idx for idx, label in enumerate(wanted_labels)}
    
    # Filter dataset to only include wanted labels
    mask = torch.tensor([label in wanted_labels for label in dataset.targets])
    dataset.data = dataset.data[mask]
    dataset.targets = dataset.targets[mask]
    
    # Map the original labels to sequential indices [0,1,2,...]
    # Convert tensor values to Python integers for dictionary lookup
    dataset.targets = torch.tensor([label_to_idx[label.item()] for label in dataset.targets])
    
    return dataset

def get_tensor_stats(tensor, counter=5):
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
    print(args.labels)

