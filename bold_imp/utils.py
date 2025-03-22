import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
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
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--thresh', type=int, default=150, metavar='N',
                        help='Threshold for the vanilla thresh boolean optimizer. A larger value means a weight will be flipped if the voting is stronger.')
    return parser.parse_args()


def filter_dataset_by_labels(dataset, wanted_labels = [1,0]):
    """Filter a dataset to only keep samples with the specified labels.
    
    Args:
        dataset: A torchvision dataset
        wanted_labels: List of labels to keep
    
    Returns:
        dataset with only samples matching wanted_labels
    """
    idx = torch.zeros_like(dataset.targets, dtype=torch.bool)
    for label in wanted_labels:
        idx |= (dataset.targets == label)
        
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]

    # Convert labels to binary (0 and 1)
    dataset.targets = (dataset.targets == wanted_labels[1]).float()
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
