# Error 

## Error1
in `/home/zlin/PIDNet/models/model_utils.py`

```
raise ValueError("Expected more than 1 value per channel when training, got input size {}".format(size))
ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 1024, 1, 1])
```

line136: in `self.scale4, BatchNorm(inplanes, momentum=bn_mom)` comment it 
or set `batchsize` > 1 


## Error2 
```
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [4, 512, 16, 16]], which is output 0 of ConstantPadNdBackward, is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
```
- solution
    - `torch.clone`
    
    ```
    out = self.conv3(out)
    out = self.norm3(out)
    out = self.rgc({0: out, 1: x[1]})
    if self.downsample is not None:
        identity = self.downsample(x[0])
    out_x = out[0].clone() + identity
    out_x = self.relu(out_x)
    out_att = out[1]
    ```
## Error3 
```
RuntimeError: Too many open files. Communication with the workers is no longer possible. Please increase the limit using `ulimit -n` in the shell or change the sharing strategy by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code
```

- solution
```
torch.multiprocessing.set_sharing_strategy('file_system')
```

## Error4
```
  File "/home/zlin/PIDNet/pidnet_env/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1146, in _try_get_data
    raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e
RuntimeError: DataLoader worker (pid(s) 595627) exited unexpectedly
```
- reduce the batch size 

## Error 5
`a tensor with 8 elements cannot be converted to scalar `
```
# x = torch.tensor([1, 2, 3]).to('cuda')
x = x.item().cpu().numpy() # so it can be written into tensorboard or logging 

# y = torch.tensor(3).to('cuda')
y = y.item() # so it can be written into tensorboard or logging 

```
## Error5 
`RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory`
- This issue happens when the file is not downloaded completely.