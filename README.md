In `bold_imp` dir, run the following:

```python3 bold_int.py --arch `cat tinyVGGadjusted` `cat args` --all-labels```

# Checkpoint
The commit `42622c28eba70255bd94a40ae6658283e9b00ed4` , under name `turn off bias by default`, contains one of the best model for MNIST. It can reach 97% in 1 epoch. Run
```python3 bold_int.py `cat args` --arch `tinyVGGadjusted` --all-labels``` for demo. Remember to switch the dataset to MNIST in args.
