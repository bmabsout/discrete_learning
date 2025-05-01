#!/usr/bin/env python3
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate arguments file with customizable parameters')
    
    # Add arguments for all the customizable parameters
    parser.add_argument('--l1-scale-min', type=float, default=-50.0, help='Minimum value for loss-int-l1-scale')
    parser.add_argument('--l1-scale-max', type=float, default=50.0, help='Maximum value for loss-int-l1-scale')
    parser.add_argument('--l1-salient', type=int, default=50, help='Value for loss-int-l1-salient')
    parser.add_argument('--act-range-min', type=float, default=-1, help='Minimum value for activation-range')
    parser.add_argument('--act-range-max', type=float, default=1, help='Maximum value for activation-range')
    parser.add_argument('--flip-ratio-decay', type=float, default=0.9, help='Value for prob-flip-ratio-decay')
    parser.add_argument('--flip-ratio-epochs', type=int, default=5, help='Value for prob-flip-ratio-decay-epochs')
    parser.add_argument('--flip-ratio-baa', type=float, default=0.3, help='Value for prob-flip-ratio-batch-acc-aware')
    parser.add_argument('--batch-size', type=int, default=384, help='Value for batch-size')
    parser.add_argument('--output-file', type=str, default='args_tmp', help='Output file name')
    
    return parser.parse_args()

def generate_args_file(args):
    # Create the template with the exact format specified
    template = f"""--dataset cifar10
--arch-custom 
--opt-momentum 
--opt-probabilistic 
--input-int 
--loss-int-l1 
--spread-std
--epochs 20 
--loss-int-l1-scale {args.l1_scale_min} {args.l1_scale_max}
--loss-int-l1-salient {args.l1_salient}
--activation-range {args.act_range_min} {args.act_range_max}
--prob-flip-ratio-decay {args.flip_ratio_decay}
--prob-flip-ratio-decay-epochs {args.flip_ratio_epochs}
--prob-flip-ratio-batch-acc-aware {args.flip_ratio_baa}
--batch-size {args.batch_size}
"""
    
    # Write the content to the output file
    with open(args.output_file, 'w') as f:
        f.write(template)
    
    print(f"Arguments file '{args.output_file}' created successfully.")

def main():
    args = parse_args()
    generate_args_file(args)

if __name__ == "__main__":
    main()

